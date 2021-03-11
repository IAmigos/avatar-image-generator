import argparse
from utils import *
from models import *

import os
import sys
import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "avatar_image_generator"

def train(config_file, use_wandb, run_name, run_notes):
    set_seed(32)
    config = configure_model(config_file, use_wandb)
    
    if use_wandb:
        wandb.init(project=PROJECT_WANDB, config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False
    
    
    xgan = Avatar_Generator_Model(config, use_wandb)
    xgan.train()


if __name__ == '__main__':
    args = parse_arguments()
    use_wandb = args.wandb
    run_name = args.run_name
    run_notes = args.run_notes

    train(CONFIG_FILENAME, use_wandb=use_wandb, run_name=run_name, run_notes=run_notes)
