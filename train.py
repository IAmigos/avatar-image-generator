import argparse
from utils import *
from models import *

import os
import sys
import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "avatar_image_generator"

def train(config_file, use_wandb=True):
    set_seed(32)

    if use_wandb:
        wandb.init(project=PROJECT_WANDB)
        wandb.watch_called = False
    
    config = configure_model(config_file, use_wandb)    
    xgan = Avatar_Generator_Model(config, use_wandb)
    xgan.train()


if __name__ == '__main__':
    #args = parse_arguments()
    use_wandb = args.wandb

    train(CONFIG_FILENAME, use_wandb=True)
