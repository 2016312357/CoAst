# -*- coding: utf-8 -*-
from copy import deepcopy
import itertools
import os
from time import time
from sklearn.utils import shuffle
import yaml
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from loguru import logger
import numpy as np
from src.models.model_utils import get_model, init_weights
from src.data.data_utils import compute_order_acc, init_wandb
import torch.nn.functional as F
from random import choices
import wandb
from utils import set_seed

QUANT_BINS = 256
DECAY_RATE = 1


def sign(x):
    return 1 if x > 0 else 0


def compute_delta(param_i, param_j, index_A, index_B):
    # param_i.cuda()
    # param_j.cuda()

    T_A_mat, T_B_mat = np.zeros((QUANT_BINS, QUANT_BINS)), np.zeros(
        (QUANT_BINS, QUANT_BINS)
    )
    T_A_i, T_A_j = np.zeros(QUANT_BINS), np.zeros(QUANT_BINS)
    T_B_i, T_B_j = np.zeros(QUANT_BINS), np.zeros(QUANT_BINS)
    set_A = set(index_A)
    set_B = set(index_B)

    set_dict_A = {}
    set_dict_B = {}
    for i in range(QUANT_BINS):
        set_dict_A[i] = set(np.argwhere(param_i == i).reshape(-1))
        set_dict_B[i] = set(np.argwhere(param_j == i).reshape(-1))

    for i, j in itertools.permutations(range(QUANT_BINS), 2):
        set_i = set_dict_A[i]
        set_j = set_dict_B[j]
        T_A_mat[i, j] = len(set_i.intersection(set_j).intersection(set_A))
        T_A_mat[i, j] /= len(index_A)
        T_B_mat[i, j] = len(set_i.intersection(set_j).intersection(set_B))
        T_B_mat[i, j] /= len(index_B)
    for i in range(QUANT_BINS):
        T_A_i[i] += len(set_dict_A[i].intersection(set_A)) / len(index_A)
        T_A_j[i] += len(set_dict_B[i].intersection(set_A)) / len(index_A)
        T_B_i[i] += len(set_dict_A[i].intersection(set_B)) / len(index_B)
        T_B_j[i] += len(set_dict_B[i].intersection(set_B)) / len(index_B)

    delta_A, delta_B = np.zeros((QUANT_BINS, QUANT_BINS)), np.zeros(
        (QUANT_BINS, QUANT_BINS)
    )
    for i, j in itertools.permutations(range(QUANT_BINS), 2):
        delta_A[i, j] = T_A_mat[i, j] - T_A_i[i] * T_A_j[j]
        delta_B[i, j] = T_B_mat[i, j] - T_B_i[i] * T_B_j[j]
    # logger.debug(delta_A.min())
    # logger.debug(delta_A.max())
    # logger.debug(delta_B.min())
    # logger.debug(delta_B.max())
    return delta_A, delta_B


@click.command()
@click.argument("config_path", type=click.STRING)
def main(config_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    # logger.info("Start FedPCA ...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check the key config
    assert "checkpoint" in config
    assert "num_classes" in config

    # Set seed
    seed = config["seed"]
    if seed == -1:
        seed = torch.randn(1, 114514)
    set_seed(seed)

    config["channel"] = 3

    rand_model = get_model(config["model_name"], config)
    rand_model = init_weights(rand_model)
    rand_model.to(device)

    # config["model_name"] = config["model_name"]

    # init_wandb(config, group=f"param_contri_partition_softmax_{config['task_type']}")

    # default config
    config["M1"] = 1000

    contribution_arr = [0] * config["num_client"]

    # logger.info("Start Loading Checkpoint")
    checkpoint_list = []
    checkpoint_list = os.listdir(f"{config['checkpoint']}/agg")
    checkpoint_list.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))

    # for filename in checkpoint_list:
    #     checkpoint_list.append(filename)

    acc_final = -1
    with torch.no_grad():
        for agg_idx, checkpoint in enumerate(checkpoint_list):
            if agg_idx < len(checkpoint_list) - 2:
                params_agg_t = f"{config['checkpoint']}/agg/{checkpoint_list[agg_idx]}"
                rand_model.load_state_dict(torch.load(params_agg_t))
                agg_params_t = torch.nn.utils.parameters_to_vector(
                    rand_model.parameters()
                )

                params_agg_t_1 = (
                    f"{config['checkpoint']}/agg/{checkpoint_list[agg_idx + 2]}"
                )
                rand_model.load_state_dict(torch.load(params_agg_t_1))
                agg_params_t_1 = torch.nn.utils.parameters_to_vector(
                    rand_model.parameters()
                )

                params_delta = agg_params_t  # agg_params_t_1 - #  check the relationship between the current aggragation and the local model

                tmp_scores = []
                for client_id in range(config["num_client"]):
                    filename = f"{config['checkpoint']}/client-{client_id}/{checkpoint_list[agg_idx]}"
                    net = torch.load(filename).to(device)
                    client_params = torch.nn.utils.parameters_to_vector(
                        net.parameters()
                    )
                    tmp_scores.append(
                        float((torch.sign(params_delta * client_params)).sum())
                    )

                # tmp_scores = np.array(tmp_scores)
                # tmp_scores = (tmp_scores - tmp_scores.min()) / (tmp_scores.max() - tmp_scores.min() + 1e-20)

                for client_id, score in enumerate(tmp_scores):
                    contribution_arr[client_id] += score * (DECAY_RATE ** (agg_idx + 1))
                # logger.debug(contribution_arr)
                acc = compute_order_acc(contribution_arr, "spearman")
                acc_final = acc
        print(f"Order-Acc: {acc_final}")
        # wandb.log({"order-acc": acc})


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
