from kornia import rgb_to_grayscale
import torch.nn as nn
import warnings, shutil
import random, textwrap
import os, glob, time
import warnings
import random, textwrap
import os, glob, time
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.utils.model_zoo
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sn
from fastai.vision.all import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
defaults.device = device

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

sn.set(style="darkgrid")
