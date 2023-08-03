# Configuration files:

## Structure:

```
DATA:
  ORIG_SIZE: Original image sizes. Evaluation exmaples are resized to this.
  SHAPE: Images sizes used for training the model.

MODEL:
  TYPE: simple / bayesian
  CHANNELS: Latent dimension.
  CH_MULT: Latent dimension multipliers per latent layer.
  NUM_RES_BLOCKS: Amount of residual blocks.
  ATTN_RESOLUTIONS: Attention resolution.
  DROPOUT: Dorpout rate.
  VAR_TYPE: fixedlarge / fixedsmall
  EMA: Use ema model or not.
  EMA_RATE: Rate of ema update.
  RESAMP_WITH_CONV: Whether to use convolutions for upsampling in UNet.

DIFFUSION:
  BETA_SCHEDULE: linear
  BETA_START: 0.0001
  BETA_END: 0.02
  NUM_DIFFUSION_TIMESTEPS: 1000
  NUM_SAMPLE_TIMESTEPS: 200  # 100
  ETA: Controls the scale of the variance (0 is DDIM, 1 is DDPM)
  SAMPLE_TYPE: generalized / ddpm_noisy
  SKIP_TYPE: uniform / quad

TRAIN:
  EPOCHS: Total number of epochs (Complete iterations through the dataset)
  BATCH_SIZE: Number of each samples in each minibatch sample
  NUM_WORKERS: Numper of threads for parallelization
  LR: Learning rate to update the model's weights
  CLIP_GRAD: Clip value for computed gradients.
  WEIGHT_DECAY: Weight decay rate / L2 regularization
  BETAS: Adam parameters
  EPS: 
  VAL_FREQ: Number of epochs between validation steps. 
  VAL_SAMPLES: Number of samples generated for validation.
```