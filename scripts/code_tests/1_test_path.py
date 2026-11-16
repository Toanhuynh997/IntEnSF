import hydra
from omegaconf import DictConfig, OmegaConf
import os 
from project_utils.paths import PROJECT_ROOT, CONFIG_DIR, DATA_DIR, CKPT_DIR


if __name__ == "__main__":
    print('PROJECT_ROOT: ', PROJECT_ROOT)
    print('CONFIG_DIR: ', CONFIG_DIR)
    print('DATA_DIR: ', DATA_DIR)
    print('CKPT_DIR: ', CKPT_DIR)
