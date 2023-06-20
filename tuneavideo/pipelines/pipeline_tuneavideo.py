# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import grid_sample
import torchvision.transforms as T

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from kornia.morphology import dilation

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange, repeat

from ..models.unet import UNet3DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(
        ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class TuneAVideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2] #* latents.shape[2] 여기에 video length저장
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1) 
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor) #shape of the video
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list): #isinstance
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def warp_latents_independently(self, latents, reference_flow):
        _, _, H, W = reference_flow.size()
        b, _, f, h, w = latents.size()
        assert b == 1
        coords0 = coords_grid(f, H, W, device=latents.device).to(latents.dtype)

        coords_t0 = coords0 + reference_flow
        coords_t0[:, 0] /= W
        coords_t0[:, 1] /= H

        coords_t0 = coords_t0 * 2.0 - 1.0

        coords_t0 = T.Resize((h, w))(coords_t0)

        coords_t0 = rearrange(coords_t0, 'f c h w -> f h w c')

        latents_0 = rearrange(latents[0], 'c f h w -> f  c  h w')
        warped = grid_sample(latents_0, coords_t0,
                             mode='nearest', padding_mode='reflection')

        warped = rearrange(warped, '(b f) c h w -> b c f h w', f=f)
        return warped
    
    def DDPM_forward(self, x0, t0, tMax, generator, device, shape, text_embeddings):
        rand_device = "cpu" if device.type == "mps" else device

        if x0 is None:
            return torch.randn(shape, generator=generator, device=rand_device, dtype=text_embeddings.dtype).to(device)
        else:
            eps = torch.randn(x0.shape, dtype=text_embeddings.dtype, generator=generator,
                              device=rand_device)
            alpha_vec = torch.prod(self.scheduler.alphas[t0:tMax])

            xt = torch.sqrt(alpha_vec) * x0 + \
                torch.sqrt(1-alpha_vec) * eps
            return xt
        
    def DDIM_backward(self, num_inference_steps, timesteps, skip_t, t0, t1, do_classifier_free_guidance, null_embs, text_embeddings, latents_local, latents_dtype, guidance_scale, guidance_stop_step, callback, callback_steps, extra_step_kwargs, num_warmup_steps):
        entered = False

        f = latents_local.shape[2]

        # latents_local = rearrange(latents_local, "b c f w h -> (b f) c w h")
    

        latents = latents_local.detach().clone()
        x_t0_1 = None
        x_t1_1 = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t > skip_t:
                    continue
                else:
                    if not entered:
                        print(
                            f"Continue DDIM with i = {i}, t = {t}, latent = {latents.shape}, device = {latents.device}, type = {latents.dtype}")
                        entered = True

                latents = latents.detach()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            
                # predict the noise residual
                with torch.no_grad():
                    if null_embs is not None:
                        text_embeddings[0] = null_embs[i][0]
                    te = torch.cat([repeat(text_embeddings[0, :, :], "c k -> f c k", f=f),
                                   repeat(text_embeddings[1, :, :], "c k -> f c k", f=f)])
                
                     # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)
                    # print("&&&&&^^^")
                    # print(text_embeddings.shape)
                    # print(te.shape)
                    # noise_pred = self.unet(
                    #     latent_model_input, t, encoder_hidden_states=te).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                if i >= guidance_stop_step * len(timesteps):
                    alpha = 0
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # latents = latents - alpha * grads / (torch.norm(grads) + 1e-10)
                # call the callback, if provided

                if i < len(timesteps)-1 and timesteps[i+1] == t0:
                    x_t0_1 = latents.detach().clone()
                    print(f"latent t0 found at i = {i}, t = {t}")
                elif i < len(timesteps)-1 and timesteps[i+1] == t1:
                    x_t1_1 = latents.detach().clone()
                    print(f"latent t1 found at i={i}, t = {t}")

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # latents = rearrange(latents, "(b f) c w h -> b c f  w h", f=f)
        res = {"x0": latents.detach().clone()}
        if x_t0_1 is not None:
            # x_t0_1 = rearrange(x_t0_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t0_1"] = x_t0_1.detach().clone()
        if x_t1_1 is not None:
            # x_t1_1 = rearrange(x_t1_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t1_1"] = x_t1_1.detach().clone()
        return res
    
    def create_motion_field(self, motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, latents):

        reference_flow = torch.zeros(
            (video_length-1, 2, 512, 512), device=latents.device, dtype=latents.dtype)
        for fr_idx, frame_id in enumerate(frame_ids):
            reference_flow[fr_idx, 0, :,
                           :] = motion_field_strength_x*(frame_id)
            reference_flow[fr_idx, 1, :,
                           :] = motion_field_strength_y*(frame_id)
        return reference_flow

    def create_motion_field_and_warp_latents(self, motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, latents):

        motion_field = self.create_motion_field(motion_field_strength_x=motion_field_strength_x,
                                                motion_field_strength_y=motion_field_strength_y, latents=latents, video_length=video_length, frame_ids=frame_ids)
        for idx, latent in enumerate(latents):
            latents[idx] = self.warp_latents_independently(
                latent[None], motion_field)
        return motion_field, latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, #*의사 난수를 생성하는 알고리즘의 상태를 관리하는 제너레이터 객체를 생성하고 반환합니다. 많은 인플레이스 무작위 샘플링 함수에서 키워드 인자로 사용됩니다.
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        guidance_stop_step: float = 0.5,
        null_embs: Optional[torch.FloatTensor] = None,
        motion_field_strength_x: float = 12,
        motion_field_strength_y: float = 12,
        xT: Optional[torch.FloatTensor] = None,
        t0: int = 44,
        t1: int = 47,
        **kwargs,
    ):
        frame_ids = kwargs.pop("frame_ids", list(range(video_length)))
        assert t0 < t1
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        timesteps_ddpm = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721,
                          701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441,
                          421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161,
                          141, 121, 101,  81,  61,  41,  21,   1]
        timesteps_ddpm.reverse()

        #* additional code 

        xT = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            1,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            xT,
        )
        dtype = xT.dtype
        
        t0 = timesteps_ddpm[t0]
        t1 = timesteps_ddpm[t1]
        
        #* Add Shape
        shape = (batch_size, num_channels_latents, 1, height //
                 self.vae_scale_factor, width // self.vae_scale_factor)
        
        ddim_res = self.DDIM_backward(num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=1000, t0=t0, t1=t1, do_classifier_free_guidance=do_classifier_free_guidance,
                                      null_embs=null_embs, text_embeddings=text_embeddings, latents_local=xT, latents_dtype=dtype, guidance_scale=guidance_scale, guidance_stop_step=guidance_stop_step,
                                      callback=callback, callback_steps=callback_steps, extra_step_kwargs=extra_step_kwargs, num_warmup_steps=num_warmup_steps)
        
        x0 = ddim_res["x0"].detach()

        if "x_t0_1" in ddim_res:
            x_t0_1 = ddim_res["x_t0_1"].detach()
        if "x_t1_1" in ddim_res:
            x_t1_1 = ddim_res["x_t1_1"].detach()
        del ddim_res
        del xT
        use_motion_field = True
        if use_motion_field:
            del x0

            x_t0_k = x_t0_1[:, :, :1, :, :].repeat(1, 1, video_length-1, 1, 1)

            reference_flow, x_t0_k = self.create_motion_field_and_warp_latents(
                motion_field_strength_x=motion_field_strength_x, motion_field_strength_y=motion_field_strength_y, latents=x_t0_k, video_length=video_length, frame_ids=frame_ids[1:])

            # assuming t0=t1=1000, if t0 = 1000
            if t1 > t0:
                x_t1_k = self.DDPM_forward(
                    x0=x_t0_k, t0=t0, tMax=t1, device=device, shape=shape, text_embeddings=text_embeddings, generator=generator)
            else:
                x_t1_k = x_t0_k

            if x_t1_1 is None:
                raise Exception

            x_t1 = torch.cat([x_t1_1, x_t1_k], dim=2).clone().detach()

            ddim_res = self.DDIM_backward(num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=t1, t0=-1, t1=-1, do_classifier_free_guidance=do_classifier_free_guidance,
                                          null_embs=null_embs, text_embeddings=text_embeddings, latents_local=x_t1, latents_dtype=dtype, guidance_scale=guidance_scale,
                                          guidance_stop_step=guidance_stop_step, callback=callback, callback_steps=callback_steps, extra_step_kwargs=extra_step_kwargs, num_warmup_steps=num_warmup_steps)

            x0 = ddim_res["x0"].detach()
            del ddim_res
            del x_t1
            del x_t1_1
            del x_t1_k
        else:
            x_t1 = x_t1_1.clone()
            x_t1_1 = x_t1_1[:, :, :1, :, :].clone()
            x_t1_k = x_t1_1[:, :, 1:, :, :].clone()
            x_t0_k = x_t0_1[:, :, 1:, :, :].clone()
            x_t0_1 = x_t0_1[:, :, :1, :, :].clone()


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
     

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return TuneAVideoPipelineOutput(videos=video)
