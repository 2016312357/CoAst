import torch
import torch.nn as nn
import torchvision
from loguru import logger
import torch.nn.functional as F
from torch.nn import init
from math import floor
from src.models.mobilevit import MobileViT
from src.models.mobilenetv2 import MobileNetV2
from src.models.dpn import TinyDPN
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation


class Swish(nn.Module):  # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


def init_weights(net, init_type="xavier", init_param=1.0):
    if init_type == "imagenet_pretrained":
        assert net.__class__.__name__ == "AlexNet"
        state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
        state_dict["classifier.6.weight"] = torch.zeros_like(net.classifier[6].weight)
        state_dict["classifier.6.bias"] = torch.ones_like(net.classifier[6].bias)
        net.load_state_dict(state_dict)
        del state_dict
        return net

    def init_func(m):
        classname = m.__class__.__name__
        if classname == "TransformerDecoder":
            for i in range(len(m.layers)):
                m.layers[i].self_attn._reset_parameters()
                m.layers[i].multihead_attn._reset_parameters()
        if classname.startswith("RNN") or classname.startswith("LSTM"):
            for names in m._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    init.constant_(bias, 0.0)
                for name in filter(lambda n: "weight" in n, names):
                    weight = getattr(m, name)
                    if init_type == "normal":
                        init.normal_(weight, 0.0, init_param)
                    elif init_type == "xavier":
                        init.xavier_normal_(weight, gain=init_param)
                    elif init_type == "xavier_unif":
                        init.xavier_uniform_(weight, gain=init_param)
                    elif init_type == "kaiming":
                        init.kaiming_normal_(weight, a=init_param, mode="fan_in")
                    elif init_type == "kaiming_out":
                        init.kaiming_normal_(weight, a=init_param, mode="fan_out")
                    elif init_type == "zero":
                        init.zeros_(weight)
                    elif init_type == "one":
                        init.ones_(weight)
                    elif init_type == "constant":
                        init.constant_(weight, init_param)
                    elif init_type == "default":
                        if hasattr(weight, "reset_parameters"):
                            weight.reset_parameters()
                    else:
                        raise NotImplementedError("initialization method [%s] is not implemented" % init_type)

        if classname.startswith("Conv") or classname == "Linear":
            if getattr(m, "bias", None) is not None:
                init.constant_(m.bias, 0.0)
            if getattr(m, "weight", None) is not None:
                if init_type == "normal":
                    init.normal_(m.weight, 0.0, init_param)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight, gain=init_param)
                elif init_type == "xavier_unif":
                    init.xavier_uniform_(m.weight, gain=init_param)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight, a=init_param, mode="fan_in")
                elif init_type == "kaiming_out":
                    init.kaiming_normal_(m.weight, a=init_param, mode="fan_out")
                elif init_type == "zero":
                    init.zeros_(m.weight)
                elif init_type == "one":
                    init.ones_(m.weight)
                elif init_type == "constant":
                    init.constant_(m.weight, init_param)
                elif init_type == "default":
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
        elif "Norm" in classname:
            if getattr(m, "weight", None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, "bias", None) is not None:
                m.bias.data.zero_()

    net.apply(init_func)
    return net


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, "relu", "instancenorm", "avgpooling"
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_model(model_name, config):
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, config["num_classes"])
        model = init_weights(model)
    elif model_name == "resnet18-nopre":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, config["num_classes"])
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(512, config["num_classes"])
    elif model_name == "bigresnet":
        model = BigResNet(
            img_size=(config["size"][0], config["size"][1]), in_channels=3, num_classes=config["num_classes"]
        )
        model = init_weights(model, init_type="xavier", init_param=1.0)
    elif model_name == "mobilenetv2-gpt":
        model = MobileNetV2(num_classes=config["num_classes"])
    elif model_name == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.last_channel, config["num_classes"])
    elif model_name == "mobilenetv2-nopre":
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.last_channel, config["num_classes"])
    elif model_name == "mobilenetv3":
        model = torchvision.models.mobilenet_v3_large(pretrained=True)
        model.classifier[-1] = nn.Linear(1280, config["num_classes"])
    elif model_name == "mobilenetv3-nobn":
        model = torchvision.models.mobilenet_v3_large(pretrained=True)
        new_features = []
        for module in model.features:
            if isinstance(module, ConvNormActivation):
                tmp = [m for m in module]
                tmp = filter(lambda m: not isinstance(m, nn.BatchNorm2d), tmp)
                new_features.append(nn.Sequential(*tmp))
            elif isinstance(module, InvertedResidual):
                # tmp = [[m for m in block] for block in module.block]
                tmp = []
                for block in module.block:
                    if isinstance(block, ConvNormActivation):
                        tmp += [m for m in block if not isinstance(m, nn.BatchNorm2d)]
                    elif isinstance(block, SqueezeExcitation):
                        tmp += [block]
                new_features.append(nn.Sequential(*tmp))
            else:
                logger.error("Special Layer in MobileNetV3 ....")
                exit(0)
        model.features = nn.Sequential(*new_features)
        model.classifier[-1] = nn.Linear(1280, config["num_classes"])
    elif model_name == "shufflenet_v2":
        model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
        last_in_features = model.fc.in_features
        model.fc = torch.nn.Linear(last_in_features, config["num_classes"])
    elif model_name == "shufflenet_v2-nopre":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
        last_in_features = model.fc.in_features
        model.fc = torch.nn.Linear(last_in_features, config["num_classes"])
    elif model_name == "squeezenet":
        model = torchvision.models.squeezenet1_1()
        final_conv = nn.Conv2d(512, config["num_classes"], kernel_size=1)
        model.classifier._modules["1"] = final_conv
        # Xaier Init the weight of model
        model = init_weights(model)
    elif model_name == "alexnet":
        model = AlexCifarNet(config)
    elif model_name == "alexnet-large":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, config["num_classes"])
    elif model_name == "tinydpn":
        model = TinyDPN(num_classes=config["num_classes"])
        model = init_weights(model)
    elif model_name == "smallresnet":
        model = SmallResNet(
            img_size=(config["size"][0], config["size"][1]), in_channels=3, num_classes=config["num_classes"]
        )
        model = init_weights(model, init_type="kaiming")
    elif model_name == "baseresnet":
        model = BaseResNet(
            img_size=(config["size"][0], config["size"][1]), in_channels=3, num_classes=config["num_classes"]
        )
        model = init_weights(model, init_type="kaiming")
    elif model_name == "convnet":
        model = ConvNet(
            channel=config["channel"],
            num_classes=config["num_classes"],
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=config["size"],
        )
    elif model_name == "mobilevit-small":
        dims = [64, 80, 96]
        channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        model = MobileViT(
            (config["size"][0], config["size"][1]), dims, channels, num_classes=config["num_classes"], expansion=2
        )
        # Xaier Init the weight of model
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.xavier_normal_(m.weight.data, 1.0)
    elif model_name == "lenet":
        model = LeNet(config)
    elif model_name == "vgg11":
        model = torchvision.models.vgg11(num_classes=config["num_classes"], pretrained=False)
        model = init_weights(model, init_type="kaiming")
        # model.fc = torch.nn.Linear(512, 10)
    elif model_name == "efficientnet-b0":
        model = torch.nn.Sequential(
            torchvision.models.efficientnet_b0(pretrained=True, num_classes=1000),
            torch.nn.AvgPool1d(4),
            torch.nn.Linear(250, config["num_classes"]),
        )
    elif model_name == "efficientnet-b0-nopre":
        model = torch.nn.Sequential(
            torchvision.models.efficientnet_b0(pretrained=False, num_classes=1000),
            torch.nn.AvgPool1d(4),
            torch.nn.Linear(250, config["num_classes"]),
        )
    elif model_name == "tiny-resnet":
        model = TinyResNet(in_channels=3, num_classes=config["num_classes"], width=config["size"][0])
    else:
        logger.error("Model Not Found!")
        exit(0)
        # return None
    return model


class AlexCifarNet(nn.Module):
    supported_dims = {32}

    def __init__(self, config):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config["channel"], 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, config["num_classes"]),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


# AlexNet66*64*3
class AlexNet(nn.Module):
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 128 * 6 * 6)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    supported_dims = {28, 32}

    def __init__(self, config):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(config["channel"], 6, 5, padding=2 if config["size"][0] == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if config["num_classes"] <= 2 else config["num_classes"])

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


class TinyResNet(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width=32):
        super(TinyResNet, self).__init__()
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2), torch.nn.ReLU()
        )
        self.max_pool_1 = torch.nn.MaxPool2d(3, 3)
        self.conv_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU()
        )
        mid_width = floor((width - 7) / 2 + 1)
        mid_width = floor((mid_width - 3) / 3 + 1)
        input_feature_num = mid_width * mid_width * 64
        self.fc = nn.Linear(input_feature_num, num_classes)

    def forward(self, x):
        x = self.max_pool_1(self.conv_1(x))
        x = x + self.conv_2(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x


class ConvNet(torch.nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == "sigmoid":
            return nn.Sigmoid()
        elif net_act == "relu":
            return nn.ReLU(inplace=True)
        elif net_act == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == "swish":
            return Swish()
        else:
            exit("unknown activation function: %s" % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == "maxpooling":
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == "avgpooling":
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == "none":
            return None
        else:
            exit("unknown net_pooling: %s" % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == "batchnorm":
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == "layernorm":
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == "instancenorm":
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == "groupnorm":
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == "none":
            return None
        else:
            exit("unknown net_norm: %s" % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != "none":
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != "none":
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


class SmallResNet(nn.Module):
    def __init__(self, img_size=(32, 32), in_channels=3, num_classes=10):
        super(SmallResNet, self).__init__()
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2), torch.nn.ReLU()
        )
        self.max_pool_1 = torch.nn.MaxPool2d(3, 3)
        self.conv_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU()
        )
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_4 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU()
        )
        tmp_feature_num = floor((img_size[0] - 2 - 7) / 2 + 1)
        tmp_feature_num = floor((tmp_feature_num - 3) / 3 + 1)
        tmp_feature_num = 128 * tmp_feature_num * tmp_feature_num
        self.fc = nn.Linear(tmp_feature_num, num_classes)

    def forward(self, x):
        x = self.max_pool_1(self.conv_1(x))
        x = x + self.conv_2(x)
        x = self.conv_3(x)
        x = x + self.conv_4(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x


class BaseResNet(nn.Module):
    def __init__(self, img_size=(32, 32), in_channels=3, num_classes=10):
        super(BaseResNet, self).__init__()
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2), torch.nn.ReLU()
        )
        self.max_pool_1 = torch.nn.MaxPool2d(3, 3)
        self.conv_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU()
        )
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_4 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU()
        )
        self.conv_5 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU()
        )
        self.conv_6 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU()
        )
        tmp_feature_num = floor((img_size[0] - 2 - 7) / 2 + 1)
        tmp_feature_num = floor((tmp_feature_num - 3) / 3 + 1)
        tmp_feature_num = 128 * tmp_feature_num * tmp_feature_num
        self.fc = nn.Linear(tmp_feature_num, num_classes)

    def forward(self, x):
        x = self.max_pool_1(self.conv_1(x))
        x = x + self.conv_2(x)
        x = self.conv_3(x)
        x = x + self.conv_4(x)
        x = x + self.conv_5(x)
        x = x + self.conv_6(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x


class BigResNet(nn.Module):
    def __init__(self, img_size=(32, 32), in_channels=3, num_classes=10):
        super(BigResNet, self).__init__()
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2), torch.nn.ReLU()
        )
        self.max_pool_1 = torch.nn.MaxPool2d(3, 3)
        self.conv_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_4 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.conv_5 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.conv_6 = torch.nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
        )
        tmp_feature_num = floor((img_size[0] - 2 - 7) / 2 + 1)
        tmp_feature_num = floor((tmp_feature_num - 3) / 3 + 1)
        tmp_feature_num = 128 * tmp_feature_num * tmp_feature_num
        self.fc = nn.Linear(tmp_feature_num, num_classes)

    def forward(self, x):
        x = self.max_pool_1(self.conv_1(x))
        x = x + self.conv_2(x)
        x = self.conv_3(x)
        x = x + self.conv_4(x)
        x = x + self.conv_5(x)
        x = x + self.conv_6(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x


if __name__ == "__main__":
    img = torch.randn(5, 3, 64, 64)
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    vit = MobileViT((64, 64), dims, channels, num_classes=100, expansion=2)
    for layer in vit.state_dict():
        print(layer, vit.state_dict()[layer].dtype)
    out = vit(img)
    print(out.shape)
