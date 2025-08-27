import os
import csv
import copy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from itertools import product
from models import get_model
from scipy.stats import ttest_1samp
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from scipy.stats import ttest_rel

def get_benign_model(args):
    benign_model = get_model(args)
    benign_model_save_path = os.path.join(args.save_path, "benign",
                                          args.dataset + "_" + args.model + "_benign_model_1.pth")
    benign_model.load_state_dict(torch.load(benign_model_save_path, map_location=args.device))
    return benign_model

def get_original_dataset(args):
    if args.dataset == "mnist":
        args.trigger_size = 3
        args.img_size = 28
        args.num_classes = 10
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                       ]))
        test_dataset = datasets.MNIST('./data', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                      ]))
    elif args.dataset == "fashionmnist":
        args.trigger_size = 3
        args.img_size = 28
        args.num_classes = 10
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                              ]))
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                             ]))
    elif args.dataset == "cifar10":
        args.trigger_size = 4
        args.img_size = 32
        args.num_classes = 10
        args.patch_size = 3
        # mean = [0.4914, 0.4822, 0.4465]
        # std = [0.2023, 0.1994, 0.2010]

        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
                
    else:
        print("wrong dataset!")
        train_dataset = None
        test_dataset = None
    return train_dataset, test_dataset

def generate_wm_samples(veri_datas, args):
    if args.trigger_name == "smile":
        trigger = torch.load("./triggers/smile.pt")
        wm_veri_datas = add_designated_trigger(veri_datas.clone(), trigger, mix_rate=args.mix_rate,
                                               position='bottom_right')
    elif args.trigger_name == "noise":
        trigger = torch.load("./triggers/noise.pt")
        wm_veri_datas = add_designated_trigger(veri_datas.clone(), trigger, mix_rate=args.mix_rate,position='bottom_right')

    elif args.trigger_name == "patch":
        wm_veri_datas = add_patch(veri_datas.clone(), patch_size=args.patch_size)

    return wm_veri_datas

def add_designated_trigger(images, trigger, mix_rate, position='bottom_right'):
    """
    add smiley trigger (tensor) on images (tensor)

    paras:
    - images: Tensor, shape [C, H, W] or [B, C, H, W]
    - trigger: Tensor, shape [C, h, w]
    - mix_rate, float, [0,1]
    - position: str, one of 'top_left', 'top_right', 'bottom_left', 'bottom_right'

    return: Tensor
    """
    if images.ndimension() == 3:
        images = images.unsqueeze(0)

    B, C, H, W = images.shape
    _, h, w = trigger.shape

    if position == 'top_left':
        y_start, x_start = 0, 0
    elif position == 'top_right':
        y_start, x_start = 0, W - w
    elif position == 'bottom_left':
        y_start, x_start = H - h, 0
    elif position == 'bottom_right':
        y_start, x_start = H - h, W - w
    else:
        raise ValueError(f"Unsupported position: {position}")

    images = images.clone()

    for i in range(B):
        images[i, :, y_start:y_start + h, x_start:x_start + w] = mix_rate * images[i, :, y_start:y_start + h, x_start:x_start + w] \
                                                                 + (1-mix_rate)*trigger

    images = torch.clamp(images,0,1)

    return images.squeeze(0) if B == 1 else images

def add_blackedge(images):
    if images.ndimension() == 3:
        images = images.unsqueeze(0)

    B, C, H, W = images.shape

    images = images.clone()

    images[:, :, 0, :] = 0  # top
    images[:, :, -1, :] = 0  # bottom
    images[:, :, :, 0] = 0  # left
    images[:, :, :, -1] = 0  # right

    images = torch.clamp(images,0,1)

    return images.squeeze(0) if B == 1 else images

def add_patch(images, patch_size):
    """
    add a black-and-white patch on images

    paras:
        images: Tensor, shape: [B, C, H, W] or [C, H, W]
        patch_size: int

    return:
        images with patch
    """
    if images.ndimension() == 3:
        images = images.unsqueeze(0)

    B, C, H, W = images.shape
    images = images.clone()

    pattern = torch.arange(patch_size).reshape(-1, 1) + torch.arange(patch_size)
    patch = (pattern % 2).float().unsqueeze(0).unsqueeze(0)

    patch = patch.expand(1, C, patch_size, patch_size)

    for i in range(B):
        images[i, :, H - patch_size:H, W - patch_size:W] = patch[0]

    images = torch.clamp(images, 0, 1)
    return images.squeeze(0) if B == 1 else images

def entropy_loss(probs):
    log_probs = torch.log(probs + 1e-12)  # proventing log(0)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def test(model, data_loader, args):
    model.eval()
    # testing
    test_loss = 0
    correct = 0

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            pred = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(pred, target, reduction='sum').item()
        # get the index of the max log-probability
        _, y_pred = torch.max(pred.data, 1)
        correct += (y_pred == target).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy

def test_backdoor(model, test_loader, args):
    model.eval()

    x_all = []
    y_all = []

    for x, y in test_loader:
        # x, y = x.to(args.device), y.to(args.device)

        mask = (y != args.target_label)
        if mask.sum() == 0:
            continue

        x_all.append(x[mask])
        y_all.append(y[mask])

    if len(x_all) == 0:
        return 0.0

    x_non_target = torch.cat(x_all, dim=0)
    y_non_target = torch.cat(y_all, dim=0)

    x_triggered = generate_wm_samples(x_non_target, args)

    with torch.no_grad():
        logits = model(x_triggered.to(args.device))
        preds = torch.argmax(logits, dim=1)

    asr_count = (preds == args.target_label).sum().item()
    total = len(y_non_target)

    attack_success_rate = asr_count / total if total > 0 else 0.0
    return 100*attack_success_rate

class TriggeredDataset(Dataset):
    def __init__(self, dataset, args,seed=42):
        self.dataset = dataset
        self.trigger_label = args.target_label
        self.patch_size = args.patch_size
        self.trigger_ratio = args.pr
        self.args = args
        self.trigger_indices = self._select_trigger_indices(seed)

    def _select_trigger_indices(self, seed):
        indices = [i for i, (_, y) in enumerate(self.dataset) if y == self.trigger_label]
        np.random.seed(seed)
        trigger_num = int(len(indices) * self.trigger_ratio)
        return set(np.random.choice(indices, trigger_num, replace=False))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        is_triggered = (idx in self.trigger_indices)
        if is_triggered:
            x = generate_wm_samples(x.unsqueeze(0), self.args).squeeze(0)
        return x, y, is_triggered

def fine_tune(model, dataloader, device, epochs):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}")
    return model

def compute_mean_activation(model, dataloader, device, target_layer_name="layer3.1.conv2"):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    target_layer = dict([*model.named_modules()])[target_layer_name]
    handle = target_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            activations.clear()
            inputs = inputs.to(device)
            _ = model(inputs)
            activation = activations[0]  # shape [B, C, H, W]
            activation = activation.mean(dim=(0, 2, 3))
            if 'activation_sum' not in locals():
                activation_sum = torch.zeros_like(activation)
            activation_sum += activation
        mean_activation = activation_sum / len(dataloader)

    handle.remove()
    return mean_activation.cpu().numpy()


def prune_channels(model, activation_scores, prune_ratio, target_layer_name="layer3.1.conv2"):
    target_layer = dict([*model.named_modules()])[target_layer_name]
    total_channels = len(activation_scores)
    num_prune = int(total_channels * prune_ratio)
    idx = np.argsort(activation_scores)[:num_prune]

    with torch.no_grad():
        target_layer.weight[idx, :, :, :] = 0.
        if target_layer.bias is not None:
            target_layer.bias[idx] = 0.
    return idx

def calaulate_pvalue(en,thre):
    t_stat, p_two_sided = ttest_1samp(en.cpu().detach().numpy(), popmean=thre)
    if t_stat < 0:
        p_value = p_two_sided / 2
    else:
        p_value = 1 - p_two_sided / 2
    return t_stat,p_value

def median_mad_threshold(values: torch.Tensor, k: float = 3.0):

    m = values.median()
    mad = (values - m).abs().median()

    if mad.item() == 0.0:
        mad = torch.tensor(1e-12, device=values.device, dtype=values.dtype)
    tau = m + k * mad
    return tau.item(), m.item(), mad.item()