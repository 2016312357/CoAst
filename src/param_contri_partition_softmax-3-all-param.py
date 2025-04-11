# -*- coding: utf-8 -*-
import os
import yaml
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from src.models.model_utils import get_model, init_weights
from src.data.data_utils import compute_order_acc
from utils import set_seed
import torch.nn.utils as utils

QUANT_BINS = 256
DECAY_RATE = 1


def sign(x):
    return 1 if x > 0 else 0


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
                agg_params_t = utils.parameters_to_vector(rand_model.parameters())

                params_agg_t_1 = (
                    f"{config['checkpoint']}/agg/{checkpoint_list[agg_idx + 2]}"
                )
                rand_model.load_state_dict(torch.load(params_agg_t_1))
                agg_params_t_1 = utils.parameters_to_vector(rand_model.parameters())

                params_delta = agg_params_t_1 - agg_params_t
                top_param_indices = torch.argsort(params_delta.abs(), descending=True)
                top_param_indices = top_param_indices[: top_param_indices.shape[0] // 2]

                tmp_scores = []
                for client_id in range(config["num_client"]):
                    filename = f"{config['checkpoint']}/client-{client_id}/{checkpoint_list[agg_idx]}"
                    net = torch.load(filename).to(device)
                    client_params = utils.parameters_to_vector(net.parameters())
                    tmp_scores.append(
                        float(
                            (
                                torch.sign(
                                    params_delta[top_param_indices]
                                    * client_params[top_param_indices]
                                )
                            ).sum()
                        )
                    )

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
