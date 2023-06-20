# BeVideo - Make your own unique video with one sentence

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

### Stable Diffusion v1-4 Model

### Training

- only 200 steps.
