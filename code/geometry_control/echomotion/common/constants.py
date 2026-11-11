import os
import numpy as np

JOINT_NUM = 24 # 22 for SMPLX
NUM_PARAMS = JOINT_NUM - 1

SMPL_MODEL_PATH = 'data/models/SMPL/SMPL_NEUTRAL.pkl'

CONFIG = {
    'smpl_model_path': SMPL_MODEL_PATH,
}