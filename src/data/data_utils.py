from typing import Dict
import torchvision
from torchvision import transforms
from loguru import logger
from copy import deepcopy
import torch
from torch.utils.data import random_split, Dataset
import numpy as np
import os
import wandb
import random
from datasets import load_dataset
from tqdm import tqdm
import pickle as pk

STATS = ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
inv_normalize = transforms.Normalize(mean=[-2.21, -2.208, -2.4], std=[5.0505, 4.975, 5.07614])

STL_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

flower_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


cifar10_train_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*STATS, inplace=True),
    ]
)

cifar10_test_transform = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(*STATS, inplace=True)]
)


stl_train_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomCrop(128, padding=4, padding_mode="reflect", fill=0),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*STL_STATS, inplace=True),
    ]
)

stl_test_transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(*STL_STATS, inplace=True)]
)


mnist_train_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


mnist_test_transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=3), transforms.Resize((32, 32)), transforms.ToTensor()]
)


food101_train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4, padding_mode="reflect", fill=0),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*STATS, inplace=True),
    ]
)

food101_test_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*STATS, inplace=True)]
)

flower_train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=4, padding_mode="reflect", fill=0),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*flower_STATS, inplace=True),
    ]
)

flower_test_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*flower_STATS, inplace=True)]
)


tiny_imagenet_train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4, padding_mode="reflect", fill=0),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*STATS, inplace=True),
    ]
)

tiny_imagenet_test_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*STATS, inplace=True)]
)


class MyTorchDataset(Dataset):
    def __init__(self, data_x, data_y) -> None:
        super().__init__()
        self.input_data = data_x
        self.targets = data_y

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index], self.targets[index]


def pre_process(x, pre_transform):
    if x.mode == "L":
        x = transforms.Grayscale(num_output_channels=3)(x)
    return pre_transform(x)


def convertToTorchDataset(hugging_dataset, pre_transform):
    data_x_arr = []
    data_y_arr = []
    for data in tqdm(hugging_dataset):
        data_x_arr.append(pre_process(data["image"], pre_transform))
        data_y_arr.append(data["label"])
    data_x_arr = torch.stack(data_x_arr)
    return MyTorchDataset(data_x_arr, data_y_arr)


def convertToTorchDatasetSimple(dataset):
    data_x_arr = [x[0] for x in dataset]
    data_y_arr = [x[1] for x in dataset]
    data_x_arr = torch.stack(data_x_arr)
    return MyTorchDataset(data_x_arr, data_y_arr)


def getDatasetConfig(dataset):
    """获取数据集的参数，以及训练的参数"""
    config = {}

    if dataset == "cifar10":
        config["num_classes"] = 10
        config["size"] = [32, 32]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        config["dataset"] = torchvision.datasets.CIFAR10(
            "./data/raw/", train=True, transform=cifar10_train_transform, download=True
        )
        config["test_dataset"] = torchvision.datasets.CIFAR10(
            "./data/raw/", train=False, transform=cifar10_test_transform, download=True
        )
    elif dataset == "stl10":
        config["num_classes"] = 10
        config["size"] = [128, 128]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        config["dataset"] = torchvision.datasets.STL10(
            "./data/raw/", split="train", transform=stl_train_transform, download=True
        )
        config["test_dataset"] = torchvision.datasets.STL10(
            "./data/raw/", split="test", transform=stl_test_transform, download=True
        )
    elif dataset == "cifar100":
        config["num_classes"] = 100
        config["size"] = [32, 32]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        config["dataset"] = torchvision.datasets.CIFAR100(
            "./data/raw/", train=True, transform=cifar10_train_transform, download=True
        )
        config["test_dataset"] = torchvision.datasets.CIFAR100(
            "./data/raw/", train=False, transform=cifar10_test_transform, download=True
        )
    elif dataset == "flower102":
        config["num_classes"] = 102
        config["size"] = [224, 224]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        if os.path.exists("./data/raw/flower102.pkl"):
            logger.info("Local DataSet is Found! Loading .....")
            config = pk.load(open("./data/raw/flower102.pkl", "rb"))
        else:
            config["dataset"] = torchvision.datasets.Flowers102(
                "./data/raw/", split="train", transform=food101_train_transform, download=True
            )
            config["test_dataset"] = torchvision.datasets.Flowers102(
                "./data/raw/", split="test", transform=food101_test_transform, download=True
            )
            config["dataset"] = convertToTorchDatasetSimple(config["dataset"])
            config["test_dataset"] = convertToTorchDatasetSimple(config["test_dataset"])
            with open("./data/raw/flower102.pkl", "wb") as f:
                pk.dump(config, f, protocol=4)
    elif dataset == "mnist":
        config["num_classes"] = 10
        config["size"] = [32, 32]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        config["dataset"] = torchvision.datasets.MNIST(
            "./data/raw/", train=True, transform=mnist_train_transform, download=True
        )
        config["test_dataset"] = torchvision.datasets.MNIST(
            "./data/raw/", train=False, transform=mnist_test_transform, download=True
        )
    elif dataset == "svhn":
        config["num_classes"] = 10
        config["size"] = [32, 32]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        full_dataset = torchvision.datasets.SVHN(
            "./data/raw/", split="train", transform=cifar10_train_transform, download=True
        )
        config["dataset"], _ = random_split(full_dataset, [50000, len(full_dataset) - 50000])
        full_test_datset = torchvision.datasets.SVHN(
            "./data/raw/", split="test", transform=cifar10_test_transform, download=True
        )
        config["test_dataset"], _ = random_split(full_test_datset, [15000, len(full_test_datset) - 15000])
    elif dataset == "food101":
        config["num_classes"] = 101
        config["size"] = [224, 224]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        if os.path.exists("./data/raw/food101.pkl"):
            logger.info("Local DataSet is Found! Loading .....")
            config = torch.load(open("./data/raw/food101.pkl", "rb"))
        else:
            config["dataset"] = torchvision.datasets.Food101(
                "./data/raw/", split="train", transform=food101_train_transform, download=True
            )
            config["test_dataset"] = torchvision.datasets.Food101(
                "./data/raw/", split="test", transform=food101_test_transform, download=True
            )
            config["dataset"] = convertToTorchDatasetSimple(config["dataset"])
            config["test_dataset"] = convertToTorchDatasetSimple(config["test_dataset"])
            with open("./data/raw/food101.pkl", "wb") as f:
                torch.save(config, f)
    elif dataset == "tiny-imagenet":
        config["num_classes"] = 200
        config["size"] = [224, 224]
        config["channel"] = 3
        config["epochs"] = 200
        config["lr"] = 1e-3
        config["optim"] = "adam"
        config["batch_size"] = 1024
        if os.path.exists("./data/raw/tiny_imagenet.pkl"):
            logger.info("Local DataSet is Found! Loading .....")
            config = pk.load(open("./data/raw/tiny_imagenet.pkl", "rb"))
        else:
            config["dataset"] = load_dataset("Maysee/tiny-imagenet", split="train")
            config["dataset"] = convertToTorchDataset(config["dataset"], tiny_imagenet_train_transform)
            config["test_dataset"] = load_dataset("Maysee/tiny-imagenet", split="valid")
            config["test_dataset"] = convertToTorchDataset(config["test_dataset"], tiny_imagenet_test_transform)
            with open("./data/raw/tiny_imagenet.pkl", "wb") as f:
                pk.dump(config, f, protocol=4)
    else:
        logger.error("No Available Config")
        exit(0)

    return config


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # logger.debug(tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


def adjust_quality(split_dataset_arr, config):
    trainset = config["dataset"]

    if config["mode"] == "noise":
        config["noise_mean_levels"] = [x * 0.01 for x in range(config["num_client"])]
        config["noise_std_levels"] = [((x / 8)) for x in range(config["num_client"])]
        for i, _ in enumerate(split_dataset_arr):
            split_dataset_arr[i].dataset = deepcopy(split_dataset_arr[i].dataset)
            transforms_list = [
                transforms.Resize((config["size"][0], config["size"][0])),
                transforms.RandomCrop(config["size"][0], padding=4, padding_mode="reflect", fill=0),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                AddGaussianNoise(config["noise_mean_levels"][i], config["noise_std_levels"][i]),
                transforms.Normalize(*STATS, inplace=True),
            ]
            if config["dataset_name"] == "mnist":
                transforms_list = [transforms.Grayscale(num_output_channels=3)] + transforms_list
            split_dataset_arr[i].dataset.transform = transforms.Compose(transforms_list)
    elif config["mode"] == "noise10":
        config["noise_mean_levels"] = [x * 0.01 for x in range(config["num_client"])]
        config["noise_std_levels"] = [((x / 32)) for x in range(config["num_client"])]
        for i, _ in enumerate(split_dataset_arr):
            split_dataset_arr[i].dataset = deepcopy(split_dataset_arr[i].dataset)
            transforms_list = [
                transforms.Resize((config["size"][0], config["size"][0])),
                transforms.RandomCrop(config["size"][0], padding=4, padding_mode="reflect", fill=0),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                AddGaussianNoise(config["noise_mean_levels"][i], config["noise_std_levels"][i]),
                transforms.Normalize(*STATS, inplace=True),
            ]
            if config["dataset_name"] == "mnist":
                transforms_list = [transforms.Grayscale(num_output_channels=3)] + transforms_list
            split_dataset_arr[i].dataset.transform = transforms.Compose(transforms_list)
    elif config["mode"] == "blur":
        config["blur_kernel_size"] = [x * 2 + 1 for x in range(config["num_client"])]
        config["blur_std"] = [((1 + 0.4 * x)) for x in range(config["num_client"])]
        for i, _ in enumerate(split_dataset_arr):
            split_dataset_arr[i].dataset = deepcopy(split_dataset_arr[i].dataset)
            transforms_list = [
                transforms.Resize((config["size"][0], config["size"][0])),
                transforms.RandomCrop(config["size"][0], padding=4, padding_mode="reflect", fill=0),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.GaussianBlur(config["blur_kernel_size"][i], config["blur_std"][i]),
                transforms.Normalize(*STATS, inplace=True),
            ]
            if config["dataset_name"] == "mnist":
                transforms_list = [transforms.Grayscale(num_output_channels=3)] + transforms_list
            split_dataset_arr[i].dataset.transform = transforms.Compose(transforms_list)
    elif config["mode"] == "count":
        split_length_arr = []
        for i in range(0, config["num_client"]):
            rest_size = int((1 - 0.1 * i) * len(split_dataset_arr[i]))
            split_length_arr.append(rest_size)
        split_length_arr.append(len(trainset) - sum(split_length_arr))
        split_dataset_arr = random_split(trainset, split_length_arr)
        split_dataset_arr = split_dataset_arr[:-1]
    elif config["mode"] == "count10":
        split_length_arr = []
        for i in range(0, config["num_client"]):
            rest_size = int((1 - 0.05 * i) * len(split_dataset_arr[i]))
            split_length_arr.append(rest_size)
        split_length_arr.append(len(trainset) - sum(split_length_arr))
        split_dataset_arr = random_split(trainset, split_length_arr)
        split_dataset_arr = split_dataset_arr[:-1]
    elif config["mode"] == "mask":
        for i, _ in enumerate(split_dataset_arr):
            split_dataset_arr[i].dataset = deepcopy(split_dataset_arr[i].dataset)
            transforms_list = [
                transforms.Resize((config["size"][0], config["size"][0])),
                transforms.RandomCrop(config["size"][0], padding=4, padding_mode="reflect", fill=0),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*STATS, inplace=True),
                transforms.RandomErasing(p=1, scale=(0.1 * i, 0.15 * i), ratio=(0.3, 3.3)),
            ]
            if config["dataset_name"] == "mnist":
                transforms_list = [transforms.Grayscale(num_output_channels=3)] + transforms_list
            split_dataset_arr[i].dataset.transform = transforms.Compose(transforms_list)
    elif config["mode"] == "mask10":
        for i, _ in enumerate(split_dataset_arr):
            split_dataset_arr[i].dataset = deepcopy(split_dataset_arr[i].dataset)
            # logger.info(f"i={i}; 0.04 * i={0.04*i}, 0.08 * i={0.08*i}")
            transforms_list = [
                transforms.Resize((config["size"][0], config["size"][1])),
                transforms.RandomCrop(config["size"][0], padding=4, padding_mode="reflect", fill=0),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*STATS, inplace=True),
                transforms.RandomErasing(p=1, scale=(0.04 * i, 0.08 * i), ratio=(0.3, 3.3)),
            ]
            if config["dataset_name"] == "mnist":
                transforms_list = [transforms.Grayscale(num_output_channels=3)] + transforms_list
            split_dataset_arr[i].dataset.transform = transforms.Compose(transforms_list)
    elif config["mode"] == "flip":
        config["wrong_label_num"] = [int(x * 0.1 * len(split_dataset_arr[x])) for x in range(config["num_client"])]

        for i, _ in enumerate(split_dataset_arr):
            split_dataset_arr[i].dataset = deepcopy(split_dataset_arr[i].dataset)
            split_dataset_arr[i].dataset.targets[: config["wrong_label_num"][i]] = [
                random.randint(0, config["num_classes"] - 1) for _ in range(config["wrong_label_num"][i])
            ]
    elif config["mode"] == "flip10":
        config["wrong_label_num"] = [int(x * 0.05 * len(split_dataset_arr[x])) for x in range(config["num_client"])]

        for i, _ in enumerate(split_dataset_arr):
            split_dataset_arr[i].dataset = deepcopy(split_dataset_arr[i].dataset)
            split_dataset_arr[i].dataset.targets[: config["wrong_label_num"][i]] = [
                random.randint(0, config["num_classes"] - 1) for _ in range(config["wrong_label_num"][i])
            ]

    elif config["mode"] == "clean":
        pass
    else:
        logger.info("No Extra Contribution Order~")

    return split_dataset_arr


def compute_order_acc(arr, mode="order"):
    n = len(arr)
    base = n * (n - 1) // 2
    ans = 0
    if mode == "order":
        for i, x in enumerate(arr):
            ans += sum([1 if x > y else 0 for y in arr[i + 1 :]])
        ans /= base
    elif mode == "spearman":
        tmp = np.argsort(arr)
        ans = 1 - (6 * sum([(n - i - 1 - x) ** 2 for i, x in enumerate(tmp)])) / (n * (n**2 - 1))
    else:
        logger.error("No Implement Score")
        exit(0)
    return ans


def getIPAddress():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    return addr


def filterConfig(config):
    newdict = {}
    for k in config:
        if isinstance(config[k], (int, float, bool, tuple, Dict, list, str)):
            newdict[k] = config[k]
    return newdict


def init_wandb(config, group="make_data", job_type="make_data"):
    """初始化wandb"""
    run_dir = "./data/log_results/"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    config["IP"] = getIPAddress()
    dataset = config["dataset"] if config.get("dataset_name") == None else config["dataset_name"]
    model_name = config["model_name"]
    client_number = config["num_client"]
    mode = config["mode"]

    wandb.init(
        config=filterConfig(config),
        project="CoAst",
        notes=f"{dataset}-{model_name}-{client_number}-{mode}",
        name=f"{dataset}-{model_name}-{client_number}-{mode}",
        group=group,
        dir=str(run_dir),
        job_type=job_type,
        reinit=True,
    )
