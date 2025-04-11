# -*- coding: utf-8 -*-
from copy import deepcopy
import os
import uuid
import yaml
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb
from src.models.model_utils import get_model, init_weights
from src.data.data_utils import adjust_quality, compute_order_acc, getDatasetConfig, init_wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from loguru import logger
import random
import numpy as np
from utils import compute_scores, set_seed
import math


def traverse_model(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            logger.info(f"name -> {name}, type -> {type(module)}, module -> {module}")
            continue
        if len(list(module.children())) > 0:
            traverse_model(module)
        else:
            logger.info(f"name -> {name}, type -> {type(module)}, module -> {module}")


def evaluate(rand_model, config, device):
    rand_model.to(device)
    rand_model.eval()

    tmp_dict = {}
    # 初始化计数器，用来计算top1，top5的正确率
    tmp_count = 0
    succ_count = 0
    succ_count_top5 = 0
    test_dataloader = DataLoader(config["test_dataset"], batch_size=config["batch_size"])
    loss_avg = torch.tensor([0.0])
    criterion = torch.nn.CrossEntropyLoss()
    for idx, (input, target) in enumerate(test_dataloader):
        out = rand_model(input.to(device))
        target = target.to(device)
        loss_avg += criterion(out, target).item()
        # 计算正确率
        for o, t in zip(out, target):
            pred = o.topk(1)[1][0]
            if pred == t:
                succ_count += 1
            if config["num_classes"] >= 5:
                preds = o.topk(5)[1]
                succ_count_top5 += 1 if t in preds else 0
            tmp_count += 1
    loss_avg /= len(config["test_dataset"]) // config["batch_size"]
    # 记录正确率
    tmp_dict["Accuracy_Top1"] = succ_count / tmp_count
    tmp_dict["Accuracy_Top5"] = succ_count_top5 / tmp_count
    tmp_dict["Test_Loss"] = loss_avg
    del test_dataloader
    return tmp_dict


@click.command()
@click.argument("config_path", type=click.STRING)
def main(config_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    # Load configs from the yaml file
    yaml_config = {}
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Set seed
    seed = yaml_config["seed"]
    if seed == -1:
        seed = torch.randn(1, 114514)
    set_seed(seed)

    # Make dir to save the models
    save_path = f"data/param_clip_models/{str(uuid.uuid4())}"
    logger.info(f"Create the dir {save_path} ....")
    os.makedirs(f"{save_path}/agg", exist_ok=True)
    for idx in range(yaml_config["num_client"]):
        os.makedirs(f"{save_path}/client-{idx}", exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the config of dataset
    config = getDatasetConfig(yaml_config["dataset"])

    # Set the config
    for k in yaml_config:
        if k == "dataset":
            config[f"{k}_name"] = yaml_config[k]
        else:
            config[k] = yaml_config[k]

    # Init Dataset with different contribution groundtruths! [Adjust the quality!]
    trainset = config["dataset"]
    train_size_per_client = len(trainset) // config["num_client"]
    logger.info(f"train_size_per_client: {train_size_per_client}")

    length_arr = [train_size_per_client] * (config["num_client"] - 1)
    length_arr.append(len(trainset) - sum(length_arr))

    split_dataset_arr = random_split(trainset, length_arr)
    split_dataset_arr = adjust_quality(split_dataset_arr, config)

    # Init Wandb
    init_wandb(
        config,
        group="train_param_clip_shapley",
        job_type=f"train_param_clip_shapley_r{int(config['param_clip_ratio']*100)}_e{config['epochs']}_s{int(config['step_size']*1000)}",
    )
    # wandb.define_metric("order_acc", summary="mean")

    # Init random model
    model_arr = []
    rand_model = get_model(config["model_name"], config)
    # rand_model.reset()
    # rand_model = init_weights(rand_model)
    rand_model.to(device)
    for _ in range(config["num_client"]):
        model_arr.append(deepcopy(rand_model))

    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0

    global_scores = [0] * config["num_client"]
    global_test_loss = None

    # BP for getting better samples
    for ep in range(config["epochs"]):
        logger.info(f"Epoch-{ep} Starting ...")

        log_dict = {}
        for client_id, train_dataset in enumerate(split_dataset_arr):
            logger.info(f"Client-{client_id} starts training ....")
            model = model_arr[client_id]
            model.to(device)
            model.train()
            dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)

            # lr_max = config["lr"]  # / config["num_client"]
            # lr_min = 1e-4
            # lr_step = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(ep / (config["epochs"] // 4) * math.pi))
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr_step, weight_decay=1e-4)

            # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

            # optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"] * ((0.5 ** (ep // 20))))
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"] / config["num_client"])

            # Backup all parameteres
            backup_params = {}
            for name in model.state_dict():
                param = model.state_dict()[name]
                backup_params[name] = deepcopy(param)

            for _ in range(config["client_epochs"]):
                loss_avg = torch.tensor([0.0])
                count = 0
                for i, (x, y) in enumerate(dataloader):
                    out = model(x.to(device))
                    loss = criterion(out, y.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_avg += loss.item()
                    count += 1
                logger.info(f"[Client-{client_id}] Loss = {loss_avg / count}")

            param_delta_list = {}

            # traverse_model(model)

            # record updated parameteres
            updated_params = {}
            for name in model.state_dict():
                param = model.state_dict()[name]
                updated_params[name] = deepcopy(param)

            for name in model.state_dict():
                # if isinstance(model._modules.get(name), torch.nn.Conv2d):
                param = model.state_dict()[name]
                param_delta_list[name] = param - backup_params[name]

            param_delta_list = [
                (layer_name, param_delta_list[layer_name].flatten()) for layer_name in param_delta_list
            ]  # ).float()

            for layer_name, d in param_delta_list:
                # logger.info(f"layer -> {layer_name}; max of d -> {torch.max(d.float())}")

                indics = torch.argsort(d.abs(), descending=True)[: int(d.shape[0] * config["param_clip_ratio"])]
                d_new = torch.zeros_like(d)

                if len(indics) == 0:
                    continue

                # method 1
                step = config["step_size"] / config["num_client"]
                d_new.index_fill_(0, indics, step)

                # method 2
                # max_value = d[indics[0]].abs()
                # min_value = d[indics[-1]].abs()

                # update_steps = min_value + (max_value - min_value) * (torch.rand(len(indics)).to(device))
                # update_steps = update_steps.sort(descending=True).values
                # d_new[indics] = update_steps

                # method 3
                # step = d[indics].abs().mean()
                # d_new.index_fill_(0, indics, step)

                # go on
                d_new *= d.sign()
                updated_params[layer_name] = backup_params[layer_name] + d_new.reshape(backup_params[layer_name].shape)

            model.load_state_dict(updated_params)  # .cpu()
            model_arr[client_id] = model.cpu()
            del model
            torch.cuda.empty_cache()
            logger.info(f"Client-{client_id} finished training ....")

        # Save all client's models
        for idx in range(config["num_client"]):
            logger.info(f"Client {idx} Save model to {save_path}/client-{idx}")
            torch.save(model_arr[idx], f"{save_path}/client-{idx}/checkpoint-{ep}.pkl")

        shapley_switch = "shapley"
        if "shapley_switch" in config and config["shapley_switch"] == "none":
            shapley_switch = "none"

        # Compute the Contribution Score
        if shapley_switch != "none":
            if global_test_loss:
                # logger.info(f"len(model_arr) -> {len(model_arr)}")
                scores = compute_scores(
                    model_arr, config["test_dataset"], global_test_loss, batch_size=config["batch_size"]
                )
                # logger.info(f"len(scores) -> {len(scores)}")
                # logger.info(f"len(global_scores) -> {len(global_scores)}")
                for i, x in enumerate(scores):
                    global_scores[i] += x
                    log_dict[f"Score-{i}"] = global_scores[i]
                log_dict["Contrib-Acc"] = compute_order_acc(global_scores, config["score_mode"])

        # Average
        state_1 = model_arr[0].state_dict()
        for layer in state_1:
            for idx in range(1, config["num_client"]):
                state_1[layer] += model_arr[idx].state_dict()[layer]
            state_1[layer] = state_1[layer].float() / config["num_client"]

        for idx in range(config["num_client"]):
            model_arr[idx].load_state_dict(deepcopy(state_1))

        # 记录正确率
        tmp_dict = evaluate(model_arr[0], config, device)
        global_test_loss = tmp_dict["Test_Loss"]
        log_dict["Accuracy_Top1"] = tmp_dict["Accuracy_Top1"]
        log_dict["Accuracy_Top5"] = tmp_dict["Accuracy_Top5"]
        if best_acc < log_dict["Accuracy_Top1"]:
            best_acc = log_dict["Accuracy_Top1"]
        logger.info(f"Best Acc: {best_acc}")
        logger.info(f"Ep-{ep}: [Top1:{log_dict['Accuracy_Top1']}, Top5:{log_dict['Accuracy_Top5']}]")
        # logger.info(f"Loss: {log_dict['loss']}")
        wandb.log(log_dict)
        logger.info(f"Save model to {save_path}/agg")
        torch.save(state_1, f"{save_path}/agg/checkpoint-{ep}.pkl")


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
