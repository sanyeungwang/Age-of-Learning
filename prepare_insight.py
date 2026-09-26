#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import math
import time
import datetime
import logging
import os
import sys
import random
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--nc', type=int, default=1024, help='#channel uses (bottleneck dim)')
    p.add_argument('--snr', type=float, default=5, help='SNR in dB for AWGN layer')
    p.add_argument('--batch', type=int, default=128, help='mini-batch size')
    p.add_argument('--epochs', type=int, default=100, help='training epochs')
    p.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    p.add_argument('--workers', type=int, default=6, help='#dataloader workers')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    p.add_argument(
        '--imb_factor',
        type=float,
        default=200,
        help='imbalance factor, e.g. 1 or 200'
    )
    return p.parse_args()


def init_logger(run_name, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s,%(msecs)03d %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info(f"Logging to {log_path}")
    return logger


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx


class ResNet34_CIFAR(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        # ---------- Stem ----------
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        # ---------- Layer1 : 3 × BasicBlock, 64 → 64 ----------
        # Block-1
        self.l1_b1_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn1 = nn.BatchNorm2d(64)
        self.l1_b1_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn2 = nn.BatchNorm2d(64)
        # Block-2
        self.l1_b2_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn1 = nn.BatchNorm2d(64)
        self.l1_b2_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn2 = nn.BatchNorm2d(64)
        # Block-3
        self.l1_b3_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b3_bn1 = nn.BatchNorm2d(64)
        self.l1_b3_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b3_bn2 = nn.BatchNorm2d(64)

        # ---------- Layer2 : 4 × BasicBlock, 64 → 128, stride=2 ----------
        # Block-1
        self.ds2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.l2_b1_conv1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)  # 32 → 16
        self.l2_b1_bn1 = nn.BatchNorm2d(128)
        self.l2_b1_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b1_bn2 = nn.BatchNorm2d(128)
        # Block-2
        self.l2_b2_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b2_bn1 = nn.BatchNorm2d(128)
        self.l2_b2_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b2_bn2 = nn.BatchNorm2d(128)
        # Block-3
        self.l2_b3_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b3_bn1 = nn.BatchNorm2d(128)
        self.l2_b3_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b3_bn2 = nn.BatchNorm2d(128)
        # Block-4
        self.l2_b4_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b4_bn1 = nn.BatchNorm2d(128)
        self.l2_b4_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b4_bn2 = nn.BatchNorm2d(128)

        # ---------- Layer3 : 6 × BasicBlock, 128 → 256, stride=2 ----------
        self.ds3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.l3_b1_conv1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)  # 16 → 8
        self.l3_b1_bn1 = nn.BatchNorm2d(256)
        self.l3_b1_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b1_bn2 = nn.BatchNorm2d(256)
        # Block-2
        self.l3_b2_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b2_bn1 = nn.BatchNorm2d(256)
        self.l3_b2_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b2_bn2 = nn.BatchNorm2d(256)
        # Block-3
        self.l3_b3_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b3_bn1 = nn.BatchNorm2d(256)
        self.l3_b3_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b3_bn2 = nn.BatchNorm2d(256)
        # Block-4
        self.l3_b4_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b4_bn1 = nn.BatchNorm2d(256)
        self.l3_b4_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b4_bn2 = nn.BatchNorm2d(256)
        # Block-5
        self.l3_b5_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b5_bn1 = nn.BatchNorm2d(256)
        self.l3_b5_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b5_bn2 = nn.BatchNorm2d(256)
        # Block-6
        self.l3_b6_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b6_bn1 = nn.BatchNorm2d(256)
        self.l3_b6_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b6_bn2 = nn.BatchNorm2d(256)

        # ---------- Layer4 : 3 × BasicBlock, 256 → 512, stride=2 ----------
        self.ds4 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.l4_b1_conv1 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)  # 8 → 4
        self.l4_b1_bn1 = nn.BatchNorm2d(512)
        self.l4_b1_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b1_bn2 = nn.BatchNorm2d(512)
        # Block-2
        self.l4_b2_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b2_bn1 = nn.BatchNorm2d(512)
        self.l4_b2_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b2_bn2 = nn.BatchNorm2d(512)
        # Block-3
        self.l4_b3_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b3_bn1 = nn.BatchNorm2d(512)
        self.l4_b3_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b3_bn2 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 512×1×1
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and name.endswith('bn2'):
                nn.init.constant_(mod.weight, 0)

    @staticmethod
    def _basic_block(x: torch.Tensor,
                     conv1: nn.Module, bn1: nn.Module,
                     conv2: nn.Module, bn2: nn.Module,
                     downsample: nn.Module = None) -> torch.Tensor:
        identity = x if downsample is None else downsample(x)
        out = F.relu(bn1(conv1(x)), inplace=True)
        out = bn2(conv2(out))
        return F.relu(out + identity, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Stem ----
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Identity

        # ---- Layer1 ----
        x = self._basic_block(x, self.l1_b1_conv1, self.l1_b1_bn1,
                              self.l1_b1_conv2, self.l1_b1_bn2)
        x = self._basic_block(x, self.l1_b2_conv1, self.l1_b2_bn1,
                              self.l1_b2_conv2, self.l1_b2_bn2)
        x = self._basic_block(x, self.l1_b3_conv1, self.l1_b3_bn1,
                              self.l1_b3_conv2, self.l1_b3_bn2)

        # ---- Layer2 ----
        x = self._basic_block(x, self.l2_b1_conv1, self.l2_b1_bn1,
                              self.l2_b1_conv2, self.l2_b1_bn2,
                              downsample=self.ds2)
        x = self._basic_block(x, self.l2_b2_conv1, self.l2_b2_bn1,
                              self.l2_b2_conv2, self.l2_b2_bn2)
        x = self._basic_block(x, self.l2_b3_conv1, self.l2_b3_bn1,
                              self.l2_b3_conv2, self.l2_b3_bn2)
        x = self._basic_block(x, self.l2_b4_conv1, self.l2_b4_bn1,
                              self.l2_b4_conv2, self.l2_b4_bn2)

        # ---- Layer3 ----
        x = self._basic_block(x, self.l3_b1_conv1, self.l3_b1_bn1,
                              self.l3_b1_conv2, self.l3_b1_bn2,
                              downsample=self.ds3)
        x = self._basic_block(x, self.l3_b2_conv1, self.l3_b2_bn1,
                              self.l3_b2_conv2, self.l3_b2_bn2)
        x = self._basic_block(x, self.l3_b3_conv1, self.l3_b3_bn1,
                              self.l3_b3_conv2, self.l3_b3_bn2)
        x = self._basic_block(x, self.l3_b4_conv1, self.l3_b4_bn1,
                              self.l3_b4_conv2, self.l3_b4_bn2)
        x = self._basic_block(x, self.l3_b5_conv1, self.l3_b5_bn1,
                              self.l3_b5_conv2, self.l3_b5_bn2)
        x = self._basic_block(x, self.l3_b6_conv1, self.l3_b6_bn1,
                              self.l3_b6_conv2, self.l3_b6_bn2)

        # ---- Layer4 ----
        x = self._basic_block(x, self.l4_b1_conv1, self.l4_b1_bn1,
                              self.l4_b1_conv2, self.l4_b1_bn2,
                              downsample=self.ds4)
        x = self._basic_block(x, self.l4_b2_conv1, self.l4_b2_bn1,
                              self.l4_b2_conv2, self.l4_b2_bn2)
        x = self._basic_block(x, self.l4_b3_conv1, self.l4_b3_bn1,
                              self.l4_b3_conv2, self.l4_b3_bn2)

        x = self.avgpool(x).flatten(1)  # (B,512)
        return self.fc(x)  # (B,num_classes)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def update_age_by_correctness(model, loader, device, age_curr):
    """
    age_curr: 1D numpy array, shape (n, )
        correct -> age=1
        wrong   -> age += 1
    """
    model.eval()
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.cpu().numpy()  # stable indices on CPU

        pred = model(x).argmax(dim=1)
        correct_mask = (pred == y).detach().cpu().numpy()  # bool

        age_curr[idx[correct_mask]] = 1
        age_curr[idx[~correct_mask]] += 1

    return age_curr


@torch.no_grad()
def per_class_accuracy(model, loader, device, num_classes=100):
    """
    Calculate the per-class accuracy on a given loader and return a NumPy array of shape (num_classes,).
    """
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    for x, y in loader:
        if isinstance(x, (list, tuple)) and len(x) == 3:
            x, y, _ = x

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x).argmax(dim=1)

        for c in range(num_classes):
            mask = (y == c)
            if mask.any():
                total[c] += mask.sum().item()
                correct[c] += (pred[mask] == c).sum().item()

    acc = correct.float() / total.clamp_min(1).float()
    return acc.cpu().numpy()


@torch.no_grad()
def per_class_margin_and_loss(model, loader, device, num_classes=100):
    """
    Compute class-level average signed logit margin and average
    unweighted softmax cross-entropy loss on the given loader.

    Per-sample margin:
        true-class logit - maximum non-true-class logit

    Returns:
        avg_margin: shape (num_classes,)
        avg_loss: shape (num_classes,)
    """
    model.eval()

    margin_sum = torch.zeros(num_classes, dtype=torch.float64)
    loss_sum = torch.zeros(num_classes, dtype=torch.float64)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        true_logits = logits.gather(
            dim=1,
            index=y.unsqueeze(1)
        ).squeeze(1)

        non_true_logits = logits.clone()
        non_true_logits.scatter_(
            1,
            y.unsqueeze(1),
            float("-inf")
        )
        max_non_true_logits = non_true_logits.max(dim=1).values

        sample_margin = true_logits - max_non_true_logits

        sample_loss = F.cross_entropy(
            logits,
            y,
            reduction="none"
        )

        y_cpu = y.detach().cpu()
        sample_margin_cpu = sample_margin.detach().cpu().double()
        sample_loss_cpu = sample_loss.detach().cpu().double()

        margin_sum.scatter_add_(
            0,
            y_cpu,
            sample_margin_cpu
        )

        loss_sum.scatter_add_(
            0,
            y_cpu,
            sample_loss_cpu
        )

        class_total.scatter_add_(
            0,
            y_cpu,
            torch.ones_like(y_cpu, dtype=torch.long)
        )

    avg_margin = (
            margin_sum / class_total.clamp_min(1).double()
    ).numpy()

    avg_loss = (
            loss_sum / class_total.clamp_min(1).double()
    ).numpy()

    return avg_margin, avg_loss


class ImbalancedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, imb_type="exp", imb_factor=100, rand_seed=42,
                 train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        if train and float(imb_factor) != 1.0:
            img_num_per_cls = self._get_img_num_per_cls(
                num_classes=100,
                imb_type=imb_type,
                imb_factor=imb_factor
            )
            self._gen_imbalanced_data(img_num_per_cls, rand_seed)

    def _get_img_num_per_cls(self, num_classes, imb_type, imb_factor):
        img_max = len(self.data) / num_classes  # CIFAR100 train: 50000/100=500
        img_num_per_cls = []
        if imb_type == "exp":
            # n_c = n_max * (1/imb_factor)^(c/(C-1))
            for c in range(num_classes):
                num = img_max * (imb_factor ** (-c / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for c in range(num_classes):
                if c < num_classes / 2:
                    img_num_per_cls.append(int(img_max))
                else:
                    img_num_per_cls.append(int(img_max / imb_factor))
        else:
            raise ValueError(f"Unknown imb_type: {imb_type}")

        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls, rand_seed):
        """
        Randomly sample each category according to img_num_per_cls to form a new long-tail training set.
        """
        np_rng = np.random.RandomState(rand_seed)

        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for cls, keep_num in zip(classes, img_num_per_cls):
            cls_idx = np.where(targets_np == cls)[0]
            np_rng.shuffle(cls_idx)
            selec_idx = cls_idx[:keep_num]

            new_data.append(self.data[selec_idx])
            new_targets.extend([cls] * keep_num)

        new_data = np.concatenate(new_data, axis=0)

        self.data = new_data
        self.targets = new_targets


def main():
    args = parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_name = f"AoI_cifar100_lt_resnet34_val_label"
    run_name = f"{ts}_{base_name}_if{int(args.imb_factor)}"
    logger = init_logger(run_name)

    args_line = ", ".join(f"{k}={v}" for k, v in vars(args).items())
    logger.info(f"Args: {args_line}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        logger.info(f'Using device {device} ({gpu_name})')
    else:
        logger.info('Using CPU')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Training set: long-tail version (with random enhancements)
    train_set = ImbalancedCIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform_train,
        imb_type="exp",
        imb_factor=args.imb_factor,  # 200/100/50/10
        rand_seed=args.seed
    )

    train_class_count = np.zeros(100, dtype=np.int64)

    if isinstance(train_set, ImbalancedCIFAR100):
        cls_counts = np.bincount(np.array(train_set.targets), minlength=100)
        logger.info(f"Imbalanced train class counts: {cls_counts.tolist()}")

        train_class_count = cls_counts.astype(np.int64)

        # logger.info("Class index -> sample count:")
        # for c, n in enumerate(cls_counts.tolist()):
        #     logger.info(f"class{c:02d}: {n}")

    test_set = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    test_targets = np.array(test_set.targets, dtype=np.int64)
    test_class_count = np.bincount(test_targets, minlength=100)

    train_eval_set = ImbalancedCIFAR100(
        root='./data',
        train=True,
        download=False,
        transform=transform_test,
        imb_type="exp",
        imb_factor=args.imb_factor,
        rand_seed=args.seed
    )

    train_eval_set = IndexedDataset(train_eval_set)

    g = torch.Generator()
    g.manual_seed(args.seed)

    def _worker_init(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        generator=g,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_worker_init
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        generator=g,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_worker_init
    )

    # new
    train_eval_loader = torch.utils.data.DataLoader(
        train_eval_set,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_worker_init
    )

    test_eval_set = IndexedDataset(test_set)
    test_eval_loader = torch.utils.data.DataLoader(
        test_eval_set,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_worker_init
    )

    model = ResNet34_CIFAR(num_classes=100).to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)

    n = len(train_eval_set)
    age_curr = np.ones(n, dtype=np.int32)  # Age=1
    age_history = []

    n_test = len(test_eval_set)
    age_curr_test = np.ones(n_test, dtype=np.int32)
    age_history_test = []

    # ======= Added: Five instantaneous statistics per epoch and per category CSV =======
    train_eval_labels = np.array(train_eval_set.base.targets, dtype=np.int64)
    num_classes = 100

    os.makedirs("per_class_stats", exist_ok=True)
    five_csv_path = os.path.join(
        "per_class_stats",
        f"{run_name}_five_per_epoch_class_stats.csv"
    )

    with open(five_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "class_idx",
            "AoL",
            "margin_train",
            "margin_test",
            "loss_train",
            "loss_test"
        ])

    best_acc = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        last_log_time = epoch_start
        running_loss = 0.0
        freq = max(1, len(train_loader) // 10)

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            lr_now = opt.param_groups[0]['lr']

            if step % freq == 0 or step == len(train_loader):
                now = time.time()
                interval = now - last_log_time
                last_log_time = now

                logger.info(
                    f"Epoch:[{epoch}/{args.epochs}] "
                    f"Iter:[{step}/{len(train_loader)}] "
                    f"Time:{interval:.2f}s "
                    f"lr:{lr_now:.4e} "
                    f"Loss:{loss.item():.4f}"
                )

        scheduler.step()

        # train_acc = evaluate(model, train_loader, device)
        train_acc = evaluate(model, train_eval_loader, device)  # changed
        test_acc = evaluate(model, test_loader, device)

        os.makedirs("weights", exist_ok=True)
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = f'weights/{run_name}_best.pth'
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved to {best_path} (acc={best_acc:.4f})")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        logger.info(
            f"Epoch:[{epoch}/{args.epochs}] "
            f"epoch_time:{epoch_time:.2f}s "
            f"avg_loss:{avg_loss:.4f} "
            f"train_acc:{train_acc:.4f} "
            f"test_acc:{test_acc:.4f} "
            f"best_acc:{best_acc:.4f}"
        )

        # Update Age on the training set (without augmentation) by "whether it is correctly classified".
        age_curr = update_age_by_correctness(
            model,
            train_eval_loader,
            device,
            age_curr
        )

        # Record the Age snapshot for the current epoch (one value per sample)
        age_history.append(age_curr.copy())

        rng_state_torch = torch.get_rng_state()

        rng_state_cuda = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        rng_state_numpy = np.random.get_state()
        rng_state_python = random.getstate()
        rng_state_loader = g.get_state()

        aol_per_class_epoch = np.full(
            num_classes,
            np.nan,
            dtype=np.float64
        )

        for c in range(num_classes):
            class_mask = (train_eval_labels == c)

            if class_mask.any():
                aol_per_class_epoch[c] = age_curr[class_mask].mean()

        margin_train_epoch, loss_train_epoch = \
            per_class_margin_and_loss(
                model,
                train_eval_loader,
                device,
                num_classes=num_classes
            )

        margin_test_epoch, loss_test_epoch = \
            per_class_margin_and_loss(
                model,
                test_loader,
                device,
                num_classes=num_classes
            )

        with open(five_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for c in range(num_classes):
                writer.writerow([
                    epoch,
                    c,
                    float(aol_per_class_epoch[c]),
                    float(margin_train_epoch[c]),
                    float(margin_test_epoch[c]),
                    float(loss_train_epoch[c]),
                    float(loss_test_epoch[c]),
                ])

        logger.info(
            f"Epoch {epoch}: five per-class instantaneous stats "
            f"appended to {five_csv_path}"
        )

        torch.set_rng_state(rng_state_torch)

        if rng_state_cuda is not None:
            torch.cuda.set_rng_state_all(rng_state_cuda)

        np.random.set_state(rng_state_numpy)
        random.setstate(rng_state_python)
        g.set_state(rng_state_loader)

        age_curr_test = update_age_by_correctness(
            model,
            test_eval_loader,
            device,
            age_curr_test
        )
        age_history_test.append(age_curr_test.copy())

        # age_history: list length = epochs; shape of each element = (n,)
        age_history_np = np.stack(age_history, axis=0)  # (E, N)

        os.makedirs("per_sample_age", exist_ok=True)

        age_csv_path = os.path.join("per_sample_age", f"{run_name}_per_sample_age.csv")
        with open(age_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header：sample_idx, epoch1, epoch2, ...
            writer.writerow(["sample_idx"] + [f"epoch{e:03d}" for e in range(1, args.epochs + 1)])

            # An age list of one sample per line
            for i in range(n):
                writer.writerow([i] + age_history_np[:, i].tolist())

        logger.info(f"Per-sample Age CSV saved to {age_csv_path}")

        # ======= (Added) Save the per-sample Age CSV for the test set =======
        age_history_test_np = np.stack(age_history_test, axis=0)  # (E, N_test)

        os.makedirs("per_sample_age_test", exist_ok=True)
        age_csv_path_test = os.path.join("per_sample_age_test", f"{run_name}_per_sample_age_test.csv")

        with open(age_csv_path_test, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx"] + [f"epoch{e:03d}" for e in range(1, args.epochs + 1)])

            for i in range(n_test):
                writer.writerow([i] + age_history_test_np[:, i].tolist())

        logger.info(f"Per-sample TEST Age CSV saved to {age_csv_path_test}")

    # ======= After training: Save the model from the last epoch =======
    last_path = f'weights/{run_name}_last.pth'
    torch.save(model.state_dict(), last_path)
    logger.info(f"Last epoch model saved to {last_path}")

    # ======= After training: Calculate per-sample and per-class average Age based on age_history =======
    # age_history: list, length = number of training epochs, each element shape=(n,)
    age_history_np = np.stack(age_history, axis=0)  # shape: (E, n)
    # Average Age of each sample over all epochs: shape (n,)
    avg_age_per_sample = age_history_np.mean(axis=0)

    labels = np.array(train_eval_set.base.targets, dtype=np.int64)
    num_classes = 100

    avg_age_per_class = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            avg_age_per_class[c] = avg_age_per_sample[mask].mean()
        else:
            avg_age_per_class[c] = np.nan

    # ======= (New) Test Set: Average Age per sample + Average Age per class =======
    age_history_test_np = np.stack(age_history_test, axis=0)  # (E, n_test)
    avg_age_per_sample_test = age_history_test_np.mean(axis=0)

    test_labels = np.array(test_set.targets, dtype=np.int64)
    avg_age_per_class_test = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = (test_labels == c)
        if mask.any():
            avg_age_per_class_test[c] = avg_age_per_sample_test[mask].mean()
        else:
            avg_age_per_class_test[c] = np.nan

    # ======= After training: Calculate the per-class accuracy of the best model and the last model on the test set =======
    if best_path is None:
        best_path = last_path
        logger.warning("best_path is None, fallback to last model as best model.")

    best_model = ResNet34_CIFAR(num_classes=100).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    per_class_acc_best = per_class_accuracy(best_model, test_loader, device, num_classes=num_classes)

    per_class_acc_last = per_class_accuracy(model, test_loader, device, num_classes=num_classes)

    per_class_avg_margin_last, per_class_avg_loss_last = \
        per_class_margin_and_loss(
            model,
            test_loader,
            device,
            num_classes=num_classes
        )

    # ======= Write a new per-class statistics CSV =======
    os.makedirs("per_class_stats", exist_ok=True)
    class_csv_path = os.path.join("per_class_stats", f"{run_name}_per_class_age_acc_counts.csv")

    with open(class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header：class_idx, avg_age, acc_best, acc_last, train_count, test_count
        writer.writerow([
            "class_idx",
            "avg_age",
            "avg_age_test",
            "per_class_acc_best",
            "per_class_acc_last",
            "per_class_avg_margin_last",
            "per_class_avg_loss_last",
            "train_class_count",
            "test_class_count"
        ])

        for c in range(num_classes):
            writer.writerow([
                c,
                float(avg_age_per_class[c]),
                float(avg_age_per_class_test[c]),
                float(per_class_acc_best[c]),
                float(per_class_acc_last[c]),
                float(per_class_avg_margin_last[c]),
                float(per_class_avg_loss_last[c]),
                int(train_class_count[c]),
                int(test_class_count[c]),
            ])

    logger.info(f"Per-class Age & accuracy stats saved to {class_csv_path}")


if __name__ == '__main__':
    main()
