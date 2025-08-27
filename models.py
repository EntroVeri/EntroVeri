import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, inception_v3, densenet121, resnet101

def get_model(args):
    if args.model == 'ResNet18':
        return get_resnet18(args)
    elif args.model == 'ResNet50':
        return get_resnet50(args)
    elif args.model == 'ResNet34':
        return get_resnet34(args)
    elif args.model == 'ResNet101':
        return get_resnet101(args)
    elif args.model == "InceptionV3":
        return inception_v3(weights=None, num_classes=args.num_classes)
    elif args.model == "DenseNet121":
        return densenet121(weights=None, num_classes=args.num_classes)
    elif args.model == "Net":
        return Net().to(args.device)
    elif args.model == "CifarNet":
        return CifarNet().to(args.device)
    elif args.model == "BetterCifarNet":
        return CifarNet().to(args.device)
    else:
        exit("Unknown Model!")

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        model = resnet18(weights=None)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        self.activ = x.clone().detach()  # 提取 layer2 输出
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        self.activ = x.clone().detach()  # same as forward
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 输出两个类（0/1）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 未加softmax，交叉熵会自动处理
        return x

#MNIST-CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        self.activ = x.clone().detach()
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.relu(self.conv1(x))
        # self.activ = x.clone().detach()
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        self.activ = x.clone().detach()
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.relu(self.conv1(x))
        self.activ = x.clone().detach()
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        return x

class BetterCifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)  # [B, 32, 32, 32]
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # [B, 64, 32, 32]
        self.pool = nn.MaxPool2d(2, 2)           # [B, 64, 16, 16]
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        self.activ = x.clone().detach()
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_resnet18(args):
    model = resnet18(weights=None, num_classes=args.num_classes)
    if args.dataset == "cifar10":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model = model.to(args.device)
    return model


def get_resnet34(args):
    model = resnet34(weights=None, num_classes=args.num_classes).to(args.device)
    return model


def get_resnet50(args):
    model = resnet50(weights=None, num_classes=args.num_classes).to(args.device)
    return model

def get_resnet101(args):
    model = resnet101(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model = model.to(args.device)
    return model