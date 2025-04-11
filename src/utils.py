import torch
import itertools
from copy import deepcopy
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np
import random


def get_subsets(arr, idx):
    subsets = []
    arr = list(set(arr) - set([idx]))
    for length in range(1, len(arr)):
        subsets += list(itertools.combinations(arr, length))
    return subsets


def compute_subset_loss(index_set, model_arr, test_dataset, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_model_arr = []
    for idx, model in enumerate(model_arr):
        if idx in index_set:
            tmp_model_arr.append(model)
    tmp_model = deepcopy(tmp_model_arr[0])
    merged_state_dict = tmp_model.state_dict()
    for layer in merged_state_dict:
        for model in tmp_model_arr[1:]:
            merged_state_dict[layer] += model.state_dict()[layer]
        merged_state_dict[layer] = merged_state_dict[layer].float() / len(tmp_model_arr)
    tmp_model.load_state_dict(merged_state_dict)

    loss_avg = torch.tensor([0.0])
    with torch.no_grad():
        tmp_model.to(device)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        criterion = torch.nn.CrossEntropyLoss()
        for x, y in test_dataloader:
            loss_avg += criterion(tmp_model(x.to(device)), y.to(device)).item()
        loss_avg /= len(test_dataset) // batch_size

    return loss_avg


def compute_scores(model_arr, test_dataset, global_test_loss, mode="shapley", batch_size=1024):
    n = len(model_arr)
    ans = [0] * n
    if mode == "shapley":
        total_index = list(range(n))
        for i, model in enumerate(model_arr):
            subsets = get_subsets(total_index, i)## all subsets without i
            tmp_score_arr = []
            for subset in subsets:
                score_1 = global_test_loss - compute_subset_loss(
                    list(subset) + [i], model_arr, test_dataset, batch_size
                )
                score_2 = global_test_loss - compute_subset_loss(subset, model_arr, test_dataset, batch_size)
                tmp_score_arr.append(score_1 - score_2)
            tmp_score = sum(tmp_score_arr) / len(tmp_score_arr)
            ans[i] = float(tmp_score)
    elif model == "none":
        pass
    else:
        logger.error("No Implement Scoring Method...")
        exit(0)
    return ans


def set_seed(SEED):
    torch.random.default_generator.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_param(model, param):
    state_dict = model.state_dict()
    for k, p in param:
        state_dict[k] = p
    model.load_state_dict(state_dict)
    return model
