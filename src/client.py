import math
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loguru import logger
from copy import deepcopy
# import torchvision
# from data_utils import inv_normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FedClient:
    pass