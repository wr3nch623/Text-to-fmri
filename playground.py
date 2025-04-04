import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
import json
import statsmodels.api as sm
from Dataset import *
from math import log

from torch.optim.lr_scheduler import StepLR, ExponentialLR

import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
import transformers

from torchviz import make_dot
import hiddenlayer as hl



loss_fn1 = nn.L1Loss()

a = torch.FloatTensor(0.255, 1.22)
b = torch.FloatTensor(0.3156, 1.34623)

print(loss_fn1(a, b))
