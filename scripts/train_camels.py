"""
Training script for CAMELS US Hydrology data using FNO.
"""

import sys
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets.camels_nc_dataset import load_camels_us_nc
from neuralop.training import setup, AdamW
from pathlib import Path
from neuralop.utils import get_wandb_api_key, count_model_params, get_project_root

# Read the configuration
from zencfg import make_config_from_cli

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
from config.camels_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

# Distributed training setup, if enabled
device, is_logger = setup(config)

# Set up WandB logging
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config.model.model_arch,
                config.model.n_layers,
                config.model.hidden_channels,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    wandb.init(**wandb_init_args)
else:
    wandb_init_args = None

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print configuration details
if config.verbose:
    print("##### CONFIG ######")
    print(config)
    sys.stdout.flush()

# Data loading setup
train_loader, test_loaders, data_processor = load_camels_us_nc(
    attr_path=config.data.attr_path,
    ts_path=config.data.ts_path,
    batch_size=config.data.batch_size,
    test_batch_size=config.data.test_batch_sizes[0],
    t_range_train=config.data.t_range_train,
    t_range_test=config.data.t_range_test,
    static_features=config.data.static_features,
    dynamic_features=config.data.dynamic_features,
    train_basins=config.data.train_basins,
    test_basins=config.data.test_basins,
)

# Set model in_channels based on data
batch = next(iter(train_loader))
sample_x = batch["x"]
config.model.data_channels = sample_x.shape[1]
if config.verbose:
    print(f"Data in_channels: {config.model.data_channels}")

# Model initialization
model = get_model(config)

# Distributed data parallel setup
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")

# Create the loss functions
l2loss = LpLoss(d=1, p=2)  # 1D time
h1loss = H1Loss(d=1)

training_loss = config.opt.training_loss
if training_loss == "l2":
    train_loss = l2loss
elif training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(f"Training_loss={training_loss} is not supported.")

eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log=config.wandb.log,
)

# Start training process
trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# Finalize WandB logging
if config.wandb.log and is_logger:
    wandb.finish()
