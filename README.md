# BeVideo - Make your own unique video with one sentence

## Topic

Video Generation from Text.
텍스트를 통해 적절한 비디오를 생성해내는 프로젝트 입니다.

### Setting

```terminal
virtualenv -p python3 {your venv name}
source {your venv name}/bin/activate
```

```python
pip install -q -U --pre triton
pip install -q diffusers[torch]==0.11.1 transformers==4.26.0 bitsandbytes==0.35.4 \
decord accelerate omegaconf einops ftfy gradio imageio-ffmpeg xformers
```
Stable Diffusion v1-4 Model 사용

### Training

- 200 steps.

## Abstract

최근 Text-to-video 생성 연구가 굉장히 활발하게 이루어지고 있습니다. Bevideo는 영상 중에서도 “영화, 드라마, MV”와 같은 영상류를 중점으로 두고 Text-to-video 작업을 진행하였습니다. 이번 프로젝트에서는 기존의 Tune-a-video (project 시작 시점 기준 가장 좋은 결과물을 가진 text to video)논문 pipeline을 개선하여 더 적합한 결과물을 만들어 내는 것을 목표로 합니다. pipeline의 중심은 diffusion model로 이루어져 있으며 핵심 수정 사항은 (1) 기존 코드 전체에 DDIM을 앞 뒤로 한 번 더 배치하여 성능을 높인 것 (2) 이미지를 동영상으로 변환하여 Text-to-video 작업을 거친 것으로 나눌 수 있습니다.

## Why?

- **Why is the problem you propose to tackle important?**
  - 텍스트로 비디오를 생성하는 주제를 선정한 이유는 상상하는 영상을 만들고 제작하는데 많은 시간과 비용이 소모되는 문제를 해결하기 위해서 입니다.
- **What impacts will it bring if the problem is addressed?**
  - 아이디어는 있지만 자본이 부족한 사람들과 영상 작업 기술이 없어 영상을 만들지 못하던 사람들이 상상한 이야기를 영상으로 만들어 낼 수 있습니다. 또한 적은 비용으로 다양한 영상을 시뮬레이션 하여 최적의 영상을 제작할 수 있습니다.
    - CG가 필요한 장면을 비롯한 영상 매체(드라마, 영화)를 만들 때 사용할 수 있습니다. 시공간의 제약을 최소화 하여 영상을 만들어 낼 수 있으며 이를 통해 천문학적인 비용을 아낄 수 있습니다. (영화 진흥 위원회에 따르면 2022년 한국 영화 개봉작 한 편당 평균 124억의 제작비가 들었습니다.)
    - 이는 광고 제작에서 굉장히 많은 효율을 낼 수 있습니다. 현재 30초의 짧은 홍보 영상 제작 비용은 크몽 (프리렌서 사이트)을 기준으로 80만원에 달합니다. 저 역시 홍보 영상을 프리렌서를 통해 제작하였는데, 이는 30초 영상에 90만원의 비용이 들었습니다. 그러나 영상의 효과는 미비했습니다. 이 프로젝트는 아직 수익이 없는 사업자들이 저렴한 비용으로 마케팅을 진행하고 광고 영상을 다양하게 테스팅 해볼 수 있도록 합니다.

## Proposed Method

1. 사진을 입력했을 때: 사진을 넣었을 때 원하는 길이의 동영상으로 변환 한 뒤, 이 동영상을 Text에 맞게 수정합니다.
2. 동영상을 입력했을 때: 원하는 Text에 맞게 동영상을 수정합니다.
3. 이 때 DDIM_Backward, DDPM_forward코드를 추가해 더 나은 결과를 도출합니다.

기존의 Tune-A-Video 코드를 개선하는 방식으로 프로젝트를 진행하였습니다.

## Previous Works

- Dreamix
  - 구현된 코드가 공개되지 않았고 논문에서 명시된 모델 및 pretrain model, data도 공개되지 않았습니다. imagen Video, Cascaded diffusion 등 여기서 사용한 모델도 공개되지 않은 모델들입니다. 그래서 이 논문의 아이디어(이미지를 Video로 바꾸는 과정)만 참고하였습니다.
  - distilled 된 모델들을 사용해 fine tune 했으나 성능이 좋지 않아서 non-distilled model 사용했다는 특징이 있습니다.
  - <img width="632" alt="스크린샷 2023-06-20 20 24 11" src="https://github.com/kains123/Bevideo/assets/48613533/46c5b964-5ae7-46d2-a3c8-9637e335f215">
  
- Tune-A-Video
  전체적인 구조는 아래와 같습니다.
  <img width="653" alt="스크린샷 2023-06-20 20 25 21" src="https://github.com/kains123/Bevideo/assets/48613533/7396c444-1b66-4752-89a7-42f4692d02a5">

  - Fintuning 부분을 보면 크게 stacked 2D convolutional residual blocks + transformer blocks로 이루어져 있고 각각의 transformer blocks들은 spatial self-atttention layer, cross-attention layer, feed forward network로 이루어져 있습니다. 맨 앞과 맨 끝은 vae를 사용해 encoding과 decoding을 진행합니다.
- Tune a video, Dreamix 서로가 서로의 방식을 디스하는 것을 논문에서 확인할 수 있었습니다…ㅎㅎ Dreamix에서는 Tune a video는 Text-to-image 모델을 기반으로 비디오를 만들어 내어 크게 모션을 수정하기 어려웠지만, 이 논문에서는 Text-to-video 모델을 사용해 좀 더 역동적으로 동작을 수정할 수 있다고 말하며, Tune a video에서는 Text-to-video 모델(VDM)을 사용했을 때의 장점이 딱히 없다는 점과 Dreamix의 pretrained model이 공개되지 않았다는 점을 지적했습니다.

etc..

- CogVideo
- Plug-and-Play
- Text2Video-Zero

> **General Editing By Video Diffusion Models**

1. Text-Guided Video Editing by Inverting Corruptions
2. Input Video Degradation
3. Text-Guided Corruption Inversion
4. Mixed Video-Image Finetuning

## **Key Model of the project**

> **Diffusion Model**

이 프로젝트에서 사용할 핵심 model은 Diffusion model 입니다.

- What is Diffusion Model?
  Diffusion models are types of Generative models that use probabilistic processes to transform data from a simple distribution to a more complex target distribution. It iteratively refines the initial random distribution, where in each step it removes the noise from the data and eventually ends up creating a realistic sample of data. This process of denoising data in each step is what is known as “Diffusion”.

  - COLAB
    [Google Colaboratory](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=xkyOEnzuVbsq)

- The reason why I chose the Diffusion model

1. Pros
   ⇒ synthesize high-quality data which includes both images and audio
   ⇒ faster to train and can use a variety of loss functions such as Wasserstein loss, and Hinge loss functions

2. Cons
   ⇒ Training a diffusion model can be a slow and computationally expensive process
   ⇒ require a large amount of data for effective training

Diffusion model을 taining 하는데 시간이 너무 많이 든다는 단점을 없애고 성능을 높이기 위해 Pretrained된 Stable Diffusion Model을 사용하였습니다. 또한 diffusion model을 사용하기 쉽게 만들어 둔 huggingface의 diffusion코드를 import하여 필요한 부분만 수정하였습니다.

[diffusers/src/diffusers at main · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/src/diffusers)

- What is Stable Diffusion?

  Stable Diffusion is based on a particular type of diffusion model called **Latent Diffusion**, proposed in [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). Diffusion models have shown to achieve state-of-the-art results for generating image data. But one downside of diffusion models is that the reverse denoising process is slow. In addition, these models consume a lot of memory because they operate in pixel space, which becomes unreasonably expensive when generating high-resolution images. Therefore, it is challenging to train these models and also use them for inference.

  [Google Colaboratory](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=Zz5Ge_47jUaA)

Stable Diffusion Model에서는 오직 CLIP trained encoder를 사용합니다. CLIP은 image encoder 와 text encoder를 사용합니다. (그래서 diffusion model을 이용한 video generate code는 대부분 CLIP을 사용합니다.)

- **Three main components in latent diffusion**

  1. **diffusers.AutoencoderKL (VAE)**

     1. `Encoder` takes an image as input and converts it into a low dimensional latent representation

     2. `Decoder` takes the latent representation and converts it back into an image
      <img width="589" alt="스크린샷 2023-06-20 20 23 28" src="https://github.com/kains123/Bevideo/assets/48613533/23ffe8d5-6bc0-4aee-a9ce-794dfc9c3a84">


     ```python
     vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
     ```

  2. **CLIPTextModel, CLIPTokenizer (Text encoder)**

     1. U-Net이 이해할 수 있게 embedding으로 transforming 해준다.

     ```python
     tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
     text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
     ```

     ```python
     uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
     text_input = pipeline.tokenizer(
             [prompt],
             padding="max_length",
             max_length=pipeline.tokenizer.model_max_length,
             truncation=True,
             return_tensors="pt",
     )

     text_embeddings = self.text_encoder(
                 text_input_ids.to(device),
                 attention_mask=attention_mask,
     )
     ```

  3. **U-Net**

     The U-Net model takes two inputs

     1. `Noisy latent or Noise`

     Noisy latents are latents produced by a VAE encoder (in case an initial image is provided) with added noise or it can take pure noise input in case we want to create a random new image based solely on a textual description

     1. `Text embeddings` - CLIP-based embedding generated by input textual prompts
        <img width="680" alt="스크린샷 2023-06-20 20 36 07" src="https://github.com/kains123/Bevideo/assets/48613533/671105db-7907-433b-885d-ba0f3f556cfd">


- **Diffusion Process**

<img width="643" alt="스크린샷 2023-06-20 20 36 28" src="https://github.com/kains123/Bevideo/assets/48613533/ec89ec00-0670-490a-a151-6aa6cd563a23">

1. The stable diffusion model은 textual input과 a seedText Input을 얻습니다. text input은 CLIP모델로 들어가서 Textual embedding of size 77x768를 만들어낸다. 그리고 seed는 1x4x64x64의 Gaussian noise를 만들어냅니다.
2. U-Net은 noise를 반복적으로 제거합니다. 이 때 text embedding을 조건으로 한다. Unet의 output은 noise가 제거된 잔차이다. Scheduler algorithm을 통해 conditional latents를 계산해냅니다.
3. 이러한 noise제거와 text conditioning은 N번 반복되며 더 나은 latent 이미지 표현을 계산해냅니다. 이 과정이 끝나면, latent image(4x64x64)를 VAE decoder가 최종 output을 뽑아냅니다.

## Dataset

- Stable Diffusion’s trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database
  [LAION-5B: A NEW ERA OF OPEN LARGE-SCALE MULTI-MODAL DATASETS | LAION](https://laion.ai/blog/laion-5b/)

## O\***\*wn inference p\*\***ipeline

- 파이프라인 설계도 그리기 (ipad로)

## Implementation

1. 사진을 입력했을 때: 사진을 넣었을 때 원하는 길이의 동영상으로 변환 한 뒤, 이 동영상을 Text에 맞게 수정할 수 있게 하였습니다.

```python
# Uplaod your image by running this cell.
#!!!!!!!!!!Choose between video and image!!!!!! #
#############[IMAGE]####################
import os
from google.colab import files
import shutil

is_static = False
uploaded = files.upload()
for filename in uploaded.keys():
    dst_path = os.path.join("data/photos", filename)
    print(dst_path)
    shutil.move(filename, dst_path)

#* Duplicate images
import shutil
for i in range(3):
  shutil.copy(f'data/photos/{filename}', f"data/photos/{i}_{filename}")

image_folder = 'data/photos'
video_name = 'video.avi'
print(filename)
v_name = filename.replace('.png', '')
v_name = filename.replace('.jpg', '')

images = [img for img in os.listdir(image_folder) if img.endswith(f"{filename}")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

convert_avi_to_mp4('video.avi', v_name)

v_name = filename.replace('.png', '.mp4')
v_name = filename.replace('.jpg', '.mp4')
shutil.move(f'{v_name}', "data/")
os.remove("video.avi")
```

```python
#is_static parameter를 추가해 noise값과 frame값을 조정해줍니다.
#그 이유는 정적 비디오에서 배워올 모션이 없기 때문입니다.

#* print pixel value
                if is_static and  pixel_values.shape[1] > 2:
                    video_length = 2;
                else :
                    video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                #* 차원 축소 b*f, c, h, w
                latents = vae.encode(pixel_values).latent_dist.sample()

                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                #* 만약 정적 비디오가 들어온다면 매우 강하게 noise를 준다.
                if is_static:
                    noise = torch.randn_like(latents, 0.7, 1)
                else :
                    noise = torch.randn_like(latents)
                bsz = latents.shape[0]

```

1. DDPM_forward, DDIM_backward를 이용해 성능을 개선합니다.

```python
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
```

```python
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
```

## Results

- **Good Video elements**

  - Alignment: 자연스럽게 연결되는 비디오를 만들어야한다.
  - Fidelity: 원래 비디오의 original input을 잘 유지해야한다.
  - Quality: 영상 자체의 퀄리티가(화질 등) 좋아야한다.

- **기존 Text to video로 Pretrained 시킨 모델로 돌린 결과물**

  A polar bear is dancing on the ice
![a polar bear is dancing on the ice](https://github.com/kains123/Bevideo/assets/48613533/1188833c-5670-4720-ac87-b30c819bb894)


- **로직 수정 후 Pretrained 시킨 모델로 돌린 결과물**

  A polar bear is dancing on the ice
![A polar bear is dancing](https://github.com/kains123/Bevideo/assets/48613533/39cb2fe0-9257-4174-8e69-5ad3e80f9e8e)


- **그 외 결과**
  a raccoon is somersault
![a raccoon is somersault](https://github.com/kains123/Bevideo/assets/48613533/ed37d617-9de6-4814-9047-eceae2698d59)


- 코드 수정 후 큰 차이는 없었지만, 사소한 오류들이 고쳐지는 것을 확인할 수 있었습니다.

## Github Links

> BeVideo Github

https://github.com/kains123/Bevideo.git


## 느낀점

- 정말 수많은 논문들을 읽어보았습니다. 이런 식으로 모델을 발전시켜 나가는 것이구나 라는 생각이 들었고 수학적 이해가 정말 중요하며 이를 완벽히 이해해야 더 나은 수식을 이끌어낼 수 있겠다 싶었습니다. 수식을 좀 더 잘 이해해보는 연습을 해야겠다는 생각을 했습니다. 어떤 식으로 수정하면 되겠다라는 생각은 머리에 떠오르는데 에러 없이 구현해내는 것이 정말 어려웠습니다.
- diffusion model을 사용한 3D nerf를 이용해 camera의 pose를 바꿔 동영상을 만들어내보고 싶습니다. 며칠 밤을 새며 시도 했으나 수많은 오류들로 끝마치지 못했습니다.

## **Future Work**

> **Future Work**
- Diffusion model로 할 수 있는 것이 정말 많다는 것을 느꼈습니다. 또한 다른 모델들과 조합한다면, 상호간의 피드백을 통해 더 정확한 결과물을 도출해낼 수 있겠다는 생각이 들었습니다. 예를 들어 objective detection을 통해 loss를 계산하는 등의 방법을 통해 서로의 모델을 상호 보완 해주는 pipeline을 구축해보고 싶습니다.
- Zero-1-to-3: Zero-shot One Image to 3D Object 또는 다른 Nerf관련 기술을 이용해 특정 이미지나 영상을 넣었을 때 촬영 구도나, camera moving을 변경할 수 있는 비디오를 생성해보고 싶습니다. 현재는 한 이미지만 넣어서는 정확한 3D modeling을 구현 하기 힘듭니다. (아래의 결과 참고) 이를 가지고 video까지 생성하기엔 상당한 어려움이 있을 것 같으나 도전해보고 싶습니다.
- https://huggingface.co/spaces/cvlab/zero123-live
<img width="640" alt="스크린샷 2023-06-20 20 39 47" src="https://github.com/kains123/Bevideo/assets/48613533/4dd8e619-f658-4fcd-b0be-3108208adc50">
