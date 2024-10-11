# %% [markdown]
# # Training
# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import torch
from youtube_asr.train import get_trainer, get_mfcc_dataloaders, upsample_factors_to_equal_label_representation
from youtube_asr.model import ClassifierLightningModule

# %%
torch.cuda.is_available()

# %%
trainer = get_trainer(
    max_epochs=3200,
    log_every_n_steps=16,
    check_val_every_n_epoch=25,
)
# %%
expand_train_factor = 1

# %%
train, val, target_names = get_mfcc_dataloaders(
    batch_size=256,
    n_mfcc=32,
    n_fft=4096,
    win_length=4096,
    hop_length=256,
    log_mels=True,
    n_mels=128,
    time_wrap=0,
    num_workers=10,
    upsample_factors= None, #{k: f * expand_train_factor for k, f in upsample_factors_to_equal_label_representation().items()},
    mask_frequency=0,
    max_frequency_mask_range=32,
    mask_time=0,
    max_time_mask_range=120,
    norm_mfcc=False,
)
# %%
model = ClassifierLightningModule(
    target_names, log_metrics_every_step=16, lr=1e-4, weight_decay=1e-4, loss='cross_entropy', label_smoothing=0.1,
)
# %%
trainer.fit(model, train, val)
# %%
