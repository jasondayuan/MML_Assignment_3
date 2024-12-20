"""
    Script that contains whole training process.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split
import sys
sys.path.append('..')

import wandb
from mml.data import ImageCaptionDataset, get_loader
from mml.model import Net, Trainer
from mml.utils import ConfigS, ConfigCustom, LRWarmup

parser = argparse.ArgumentParser()

parser.add_argument(
    "-C", "--checkpoint-name", type=str, default="", help="Checkpoint name"
)

parser.add_argument(
    "-S",
    "--size",
    type=str,
    default="S",
    choices=["S", "C", "s", "c"],
)

args = parser.parse_args()

config = ConfigCustom() if args.size.upper() == "C" else ConfigS()

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    # TODO: 需要你自己实现一个ImageCaptionDataset在`data/dataset.py`中
    data_root_dir = "/scratch/jason/workspace/mml/mml-assignment3-main/mml/data/coco" 
    dataset = ImageCaptionDataset(config.clip_model, root_dir=data_root_dir)

    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [config.train_size, config.val_size, config.test_size]
    )

    train_loader = get_loader(
        train_dataset,
        bs_exp=config.batch_size_exp if is_cuda else 2,
        shuffle=True,
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
        text_model=config.text_model,
    )

    valid_loader = get_loader(
        val_dataset,
        bs_exp=config.batch_size_exp if is_cuda else 2,
        shuffle=False,
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
        text_model=config.text_model,
    )

    model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()

    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_dataset=test_dataset,
        test_path=os.path.join(data_root_dir, "train2014"), # TODO: 请你修改这里为你自己的目录
        ckp_path=ckp_path,
        device=device,
    )

    # build train model process with experiment tracking from wandb
    wandb.init(project="captioner", config=config.__dict__)
    wandb.watch(trainer.model, log="all")
    for epoch in range(trainer.epoch, config.epochs):
        trainer.train_epoch()
        trainer.valid_epoch()
        trainer.test_step()

        metadata = trainer.get_training_data()

        # log loss to wandb
        wandb.log(
            {
                "train_loss/loss": metadata["train_loss"][-1],
                "valid_loss/loss": metadata["valid_loss"][-1],
                "lr": metadata["lr"],
                "examples": wandb.Image(metadata["examples"]),
            }
        )

        if not os.path.exists(config.weights_dir):
            os.makedirs(config.weights_dir)

        if (epoch + 1) % 6 == 0:
            trainer.save_ckp(os.path.join(config.weights_dir, f"epoch_{epoch + 1}.pt"))
