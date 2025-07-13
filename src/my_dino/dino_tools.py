"""
From Concise Dino
"""
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

def simple_imshow_tensor(im):
    # Assuming `img` is your image tensor with shape [3, 64, 64]
    im = im.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    im = (im-im.min())/(im.max()-im.min())
    # If your image values are in [0, 255], you can normalize them to [0, 1]
    # im = im / 255.0
    plt.imshow(im)
    plt.axis('off')  # Hide axes for better visualization
    plt.show

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True