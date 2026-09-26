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
import io
import struct
import lmdb
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import VisionTransformer

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--nc', type=int, default=1024, help='#channel uses (bottleneck dim)')
    p.add_argument('--snr', type=float, default=5, help='SNR in dB for AWGN layer')
    p.add_argument('--batch', type=int, default=128, help='mini-batch size')
    p.add_argument('--eval_batch', type=int, default=100, help='eval batch size (official default=100)')
    p.add_argument('--epochs', type=int, default=200, help='training epochs')
    p.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    p.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (official default=0.9)')
    p.add_argument('--weight_decay', type=float, default=2e-4, help='L2 weight decay (official default=2e-4)')
    p.add_argument('--workers', type=int, default=6, help='#dataloader workers')
    p.add_argument('--seed', type=int, default=42, help='random seed')

    p.add_argument('--model', type=str, default='resnet32', choices=['resnet32', 'resnet34', 'cifar_vit'],
                   help='model to use')
    p.add_argument('--dataset', type=str, default='cifar100',
                   choices=['cifar10', 'cifar100', 'imagenet', 'places365', 'imagenet_subset', 'places365_subset'],
                   help='dataset to use')
    p.add_argument('--data_root', type=str, default='./data_large',
                   help='root directory containing imagenet_subset and places365_subset')
    p.add_argument('--imb_factor', type=float, default=200, help='imbalance factor 1, 10, 20, 50, 100, 200, 300, 400')
    p.add_argument('--epoch_drop_percent', type=float, default=0,
                   help='percentage of training samples randomly dropped at the start of each epoch (0-100); 0 disable')
    p.add_argument('--setting', type=str, default='offline', choices=['offline', 'online', 'online_dynamic_lt'],
                   help='training setting: offline (existing logic) or online (new validation-buffer logic)')
    p.add_argument('--buffer_mode', type=str, default='infinite', choices=['infinite', 'fixed'],
                   help='validation buffer mode')
    p.add_argument('--buffer_size', type=int, default=10000, help='max size for fixed buffer')
    p.add_argument('--lt_shuffle_period', type=int, default=1,
                   help='For online_dynamic_lt only: change head/tail class order every K epochs (K>=1). '
                        'K=1 means reshuffle every epoch.')

    p.add_argument('--age_fuse', type=str, default='ema', choices=['ema', 'avg'],
                   help="How to fuse historical per-sample age for reweighting: "
                        "'ema' uses exponential moving average with --age_rho; "
                        "'avg' uses simple mean over all past epochs (age_avg).")
    p.add_argument('--age_rho', type=float, default=0,
                   help='EMA factor rho in [0,1] for age-based reweighting (0=standard CE, 1=use latest age only)')
    p.add_argument('--cb_beta', type=float, default=0.9,
                   help='CB beta in [0,1). Paper searches {0.9,0.99,0.999,0.9999}')
    p.add_argument('--focal_gamma', type=float, default=1.0, help='focal gamma, e.g. 0.5/1.0/2.0')

    p.add_argument('--loss_type', type=str, default='softmax', choices=['softmax', 'sigmoid', 'focal'],
                   help='loss type: softmax CE (4.1), sigmoid CE (4.2), focal (4.3)')
    p.add_argument('--weight_scope', type=str, default='class', choices=['class', 'sample'],
                   help='Where age-based weighting is applied: class-level or sample-level.')

    p.add_argument('--loss_weight_type', type=str, default='none', choices=['none', 'cb', 'age', 'cb_age'],
                   help='loss weight type for pure CE: none/cb/age/cb_age')
    p.add_argument('--sampler_weight_type', type=str, default='none', choices=['none', 'age', 'cb'],
                   help='Sampler weight type: none, age (Age-RS), or cb (CB-RS).')

    # Only effective when --model cifar_vit is used
    p.add_argument('--vit_patch_size', type=int, default=4,
                   help='ViT patch size for 32x32 CIFAR images')
    p.add_argument('--vit_embed_dim', type=int, default=192,
                   help='ViT embedding dimension')
    p.add_argument('--vit_depth', type=int, default=6,
                   help='ViT encoder depth')
    p.add_argument('--vit_num_heads', type=int, default=3,
                   help='ViT number of attention heads')
    p.add_argument('--vit_mlp_dim', type=int, default=768,
                   help='ViT MLP hidden dimension')
    p.add_argument('--vit_dropout', type=float, default=0.1,
                   help='ViT dropout')
    p.add_argument('--vit_lr', type=float, default=3e-4,
                   help='ViT learning rate')
    p.add_argument('--vit_weight_decay', type=float, default=5e-2,
                   help='ViT weight decay')
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


class LMDBImageDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, targets_path, transform=None):
        self.lmdb_path = lmdb_path
        self.targets = np.load(targets_path).astype(np.int64)
        self.transform = transform
        self.env = None

    def __len__(self):
        return len(self.targets)

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
        return self.env

    def __getitem__(self, idx):
        env = self._get_env()

        key = struct.pack(">Q", int(idx))

        with env.begin(write=False) as txn:
            encoded = txn.get(key)

        if encoded is None:
            raise KeyError(f"LMDB key not found: {idx}")

        label = struct.unpack(">H", encoded[:2])[0]
        image_bytes = encoded[2:]

        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return image, int(label)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state


def build_epoch_train_subset(base_dataset, drop_percent, seed, epoch):
    """
    At the start of each epoch, randomly drop drop_percent% of the samples from base_dataset,
    and return a Subset containing the remaining samples together with the corresponding original indices keep_indices.

    drop_percent = 0.0 -> fully falls back to the original pipeline
    """
    n = len(base_dataset)

    if drop_percent < 0.0 or drop_percent > 100.0:
        raise ValueError(f"--epoch_drop_percent must be in [0,100], got {drop_percent}")

    if drop_percent == 0.0:
        keep_indices = np.arange(n, dtype=np.int64)
    else:
        keep_num = int(round(n * (100.0 - drop_percent) / 100.0))
        # Prevent the set from becoming empty when 100% is dropped so training can continue
        keep_num = max(1, keep_num)

        rng = np.random.default_rng(seed + epoch)
        keep_indices = rng.choice(n, size=keep_num, replace=False).astype(np.int64)

    epoch_train_set = torch.utils.data.Subset(base_dataset, keep_indices.tolist())
    return epoch_train_set, keep_indices


def get_img_num_per_cls(num_classes, img_max, imb_type, imb_factor):
    """
    Generate the long-tail per-class sample-count template from the head-class maximum sample count img_max.
    Return a numpy array of length num_classes, ordered by rank 0..C-1 from head to tail by default.
    """
    if imb_factor < 1:
        raise ValueError(f"imb_factor must be >= 1, got {imb_factor}")

    img_num_per_cls = []

    if imb_type == "exp":
        for c in range(num_classes):
            num = img_max * (imb_factor ** (-c / (num_classes - 1.0)))
            img_num_per_cls.append(max(1, int(num)))
    elif imb_type == "step":
        for c in range(num_classes):
            if c < num_classes / 2:
                img_num_per_cls.append(int(img_max))
            else:
                img_num_per_cls.append(max(1, int(img_max / imb_factor)))
    else:
        raise ValueError(f"Unknown imb_type: {imb_type}")

    return np.asarray(img_num_per_cls, dtype=np.int64)


def build_epoch_dynamic_longtail_subset(
        base_dataset,
        labels,
        num_classes,
        imb_factor,
        seed,
        epoch,
        imb_type="exp",
        rank_perm=None,
        class_indices=None,
):
    """
    Dynamically construct a long-tail training subset for the current epoch from a balanced source pool.

    Key properties:
    1) the source pool is a balanced IF=1 dataset
    2) each epoch is resampled
    3) each epoch randomly shuffles which class is head / second-head / ... / tail
    4) this subset is what actually goes into training

    Returns:
        epoch_train_set: Subset(base_dataset, keep_indices)
        keep_indices: shape (N_keep,)
        target_counts_by_class: shape (C,) how many samples each real class is finally drawn for in the current epoch
        rank_perm: shape (C,)
            rank_perm[r] = the real class id corresponding to rank r in the long-tail ordering
            e.g., rank_perm[0] is the head class in this round
    """
    labels = np.asarray(labels, dtype=np.int64)

    # Indices of each real class in the source pool.
    # Keep the original behavior unless precomputed indices are explicitly supplied.
    if class_indices is None:
        class_indices = [
            np.where(labels == c)[0]
            for c in range(num_classes)
        ]

    class_sizes = np.asarray(
        [len(v) for v in class_indices],
        dtype=np.int64
    )

    # Use the minimum class size as img_max for stability; on IF=1 CIFAR this is 500 / 5000 per class
    img_max = int(class_sizes.min())

    # Standard long-tail template: rank from head -> tail
    template_counts = get_img_num_per_cls(
        num_classes=num_classes,
        img_max=img_max,
        imb_type=imb_type,
        imb_factor=imb_factor,
    )  # shape (C,)

    rng = np.random.default_rng(seed + epoch)

    # Randomly decide which real class gets rank 1 (head), rank 2, ..., and rank C (tail) in this round
    # If not specified, resample a rank_perm for this epoch;
    # If specified, reuse the externally provided class order
    if rank_perm is None:
        rank_perm = rng.permutation(num_classes)  # shape (C,)
    else:
        rank_perm = np.asarray(rank_perm, dtype=np.int64)
        if rank_perm.shape != (num_classes,):
            raise ValueError(
                f"rank_perm shape mismatch: got {rank_perm.shape}, expected ({num_classes},)"
            )
        if len(np.unique(rank_perm)) != num_classes:
            raise ValueError("rank_perm must be a permutation of class ids")

    # target_counts_by_class[real_class] = how many samples this real class should receive in this round
    target_counts_by_class = np.zeros(num_classes, dtype=np.int64)
    target_counts_by_class[rank_perm] = template_counts

    keep_list = []
    for cls_id in range(num_classes):
        cls_pool = class_indices[cls_id]
        need = int(target_counts_by_class[cls_id])

        if need > len(cls_pool):
            raise ValueError(
                f"class {cls_id}: need {need} samples, but source pool only has {len(cls_pool)}"
            )

        chosen = rng.choice(cls_pool, size=need, replace=False)
        keep_list.append(chosen)

    keep_indices = np.concatenate(keep_list, axis=0).astype(np.int64)

    # Shuffle the final subset indices
    rng.shuffle(keep_indices)

    epoch_train_set = torch.utils.data.Subset(base_dataset, keep_indices.tolist())
    return epoch_train_set, keep_indices, target_counts_by_class, rank_perm


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
    loader: must return (x, y, idx), and shuffle=False
    Rules:
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

        # Vectorized update
        age_curr[idx[correct_mask]] = 1
        age_curr[idx[~correct_mask]] += 1

    return age_curr


@torch.no_grad()
def update_age_with_access_constraint(
        model,
        access_loader,
        device,
        age_curr,
        prev_seen_mask,
        current_mask,
):
    """
    Rules:
    - Samples never seen before: keep NaN, do not count them
    - Samples entering the current epoch for the first time: initialize age=1
    - Previously seen but absent in the current epoch: age += 1 and do not participate in prediction
    - Samples appearing in the current epoch: participate in prediction
        correct -> age = 1
        wrong   -> age += 1
    """
    model.eval()
    age_next = age_curr.copy()

    # 1) Samples seen for the first time in this epoch: initialize age=1
    new_seen_mask = current_mask & (~prev_seen_mask)
    age_next[new_seen_mask] = 1.0

    # 2) Previously seen, but absent in this epoch: directly add 1 without prediction
    old_missing_mask = prev_seen_mask & (~current_mask)
    age_next[old_missing_mask] = age_next[old_missing_mask] + 1.0

    # 3) Only predict on samples that appear in this epoch
    for x, y, idx in access_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.cpu().numpy()

        pred = model(x).argmax(dim=1)
        correct_mask = (pred == y).detach().cpu().numpy()

        # Prediction correct: reset directly to 1
        age_next[idx[correct_mask]] = 1.0

        # Prediction wrong: add 1 to the current value
        age_next[idx[~correct_mask]] = age_next[idx[~correct_mask]] + 1.0

    return age_next


@torch.no_grad()
def per_class_accuracy(model, loader, device, num_classes):
    """
    Compute per-class accuracy on the given loader, returning a numpy array of shape (num_classes,)
    """
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x).argmax(dim=1)

        for c in range(num_classes):
            mask = (y == c)
            if mask.any():
                total[c] += mask.sum().item()
                correct[c] += (pred[mask] == c).sum().item()

    acc = correct.float() / total.clamp_min(1).float()
    return acc.cpu().numpy()  # numpy array of length num_classes


@torch.no_grad()
def per_class_margin_and_loss(model, loader, device, num_classes):
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

        # True-class logits
        true_logits = logits.gather(
            dim=1,
            index=y.unsqueeze(1)
        ).squeeze(1)

        # Maximum logit among all non-true classes
        non_true_logits = logits.clone()
        non_true_logits.scatter_(
            1,
            y.unsqueeze(1),
            float("-inf")
        )
        max_non_true_logits = non_true_logits.max(dim=1).values

        # Signed logit margin
        sample_margin = true_logits - max_non_true_logits

        # Plain per-sample softmax CE
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


@torch.no_grad()
def compute_class_stats(model, loader, device, num_classes):
    """
        Purpose:
            On the current validation buffer, compute for each class:
            - number of correct predictions class_correct[c]
            - total number of samples class_total[c]
            - accuracy class_acc[c]

        This is the statistics stage in the online setting; it only performs forward passes and does not update any state.

        Returns:
            class_acc: shape (C,) per-class accuracy
            class_correct: shape (C,) number of correct predictions per class
            class_total: shape (C,) number of samples per class
    """
    model.eval()

    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x).argmax(dim=1)

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        for c in range(num_classes):
            mask = (y_np == c)
            if mask.any():
                class_total[c] += int(mask.sum())
                class_correct[c] += int((pred_np[mask] == c).sum())

    class_acc = np.zeros(num_classes, dtype=np.float64)
    valid_mask = class_total > 0
    class_acc[valid_mask] = class_correct[valid_mask] / class_total[valid_mask]

    return class_acc, class_correct, class_total


def update_class_age_with_threshold(
        class_age,
        class_acc,
        class_total,
        acc_threshold,
):
    """
    Purpose:
        Update class-level age according to the current per-class accuracy and the threshold.

    Rule (core algorithm):
        For classes that appear in the current buffer:

            if acc[c] < threshold:
                class_age[c] += 1      # Hard classes -> age increases -> weight increases

            else:
                class_age[c] = 1       # Easy classes -> reset age

        For classes that never appear:
            Do not update (keep the previous value)

    Returns:
        class_age_next: updated age
    """
    class_age_next = class_age.copy()

    valid_mask = class_total > 0
    bad_mask = valid_mask & (class_acc < acc_threshold)
    good_mask = valid_mask & (class_acc >= acc_threshold)

    class_age_next[bad_mask] += 1.0
    class_age_next[good_mask] = 1.0

    return class_age_next


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    # logits: (B, C)
    return torch.sigmoid(logits)


def softmax_probs(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # logits: (B, C)
    return torch.softmax(logits, dim=dim)


def sigmoid_ce_loss(
        logits: torch.Tensor,  # (B, C)
        target: torch.Tensor,  # (B,)
        weight_bc: torch.Tensor | None = None  # (B, C) or None
) -> torch.Tensor:
    """
    Pure sigmoid CE (without the focal modulator), used to reproduce Section 4.2 on its own (together with CB weight_bc).
    """
    _, num_classes = logits.shape
    one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=logits.dtype)  # (B, C)

    # BCEWithLogits per entry
    bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction='none')  # (B, C)

    if weight_bc is not None:
        bce = bce * weight_bc  # (B, C)

    # In TF this is sum / sum(one_hot); for single-label classification, sum(one_hot)=B
    return bce.sum() / one_hot.sum().clamp_min(1.0)


def focal_loss(
        logits: torch.Tensor,  # (B, C)
        target: torch.Tensor,  # (B,)
        gamma: float,
        weight_bc: torch.Tensor | None = None  # (B, C) or None (passed when using CB)
) -> torch.Tensor:
    """
    Pure focal loss (does not generate CB weights), can be wrapped by CB focal.
    """
    _, num_classes = logits.shape
    one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=logits.dtype)  # (B,C)

    # cross_entropy = sigmoid_cross_entropy_with_logits(labels, logits)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, one_hot, reduction='none'
    )  # (B,C)

    if gamma == 0.0:
        modulator = torch.ones_like(logits)
    else:
        # TF: exp(-gamma * y * logit - gamma * log1p(exp(-logit)))
        modulator = torch.exp(
            -gamma * one_hot * logits - gamma * F.softplus(-logits)
        )  # (B,C)

    loss = modulator * cross_entropy

    if weight_bc is not None:
        loss = loss * weight_bc

    # Same normalization as sigmoid CE (single-label -> divide by B)
    return loss.sum() / one_hot.sum().clamp_min(1.0)


def cb_softmax_ce_loss(logits, target, class_w):  # 4.1
    batch_size, num_classes = logits.shape
    one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=logits.dtype)
    sample_weight = (one_hot * class_w.view(1, -1)).sum(dim=1)  # (B,)
    ce = F.cross_entropy(logits, target, reduction='none')
    return (sample_weight * ce).sum() / sample_weight.sum().clamp_min(1e-12)


def cb_sigmoid_ce_loss(logits, target, class_w):  # 4.2
    batch_size, num_classes = logits.shape
    one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=logits.dtype)  # (B,C)
    # sample_weight = sum_c w[c]*one_hot => (B,)
    sample_weight = (one_hot * class_w.view(1, -1)).sum(dim=1, keepdim=True)  # (B,1)
    weight_bc = sample_weight.expand(batch_size, num_classes)  # (B,C)
    return sigmoid_ce_loss(logits, target, weight_bc=weight_bc)


def cb_focal_loss(logits, target, class_w, gamma):  # 4.3
    batch_size, num_classes = logits.shape
    # one_hot: (B, C)
    one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=logits.dtype)
    # sample_weight: (B, 1) = sum_c one_hot[b,c] * class_w[c]
    sample_weight = (one_hot * class_w.view(1, -1)).sum(dim=1, keepdim=True)
    weight_bc = sample_weight.expand(batch_size, num_classes)
    return focal_loss(logits, target, gamma=gamma, weight_bc=weight_bc)


# new for loss 6
def compute_class_weights_from_age_history(
        age_hist_list,
        y_all,
        n_cls,
        rho=0.5,
        fuse='ema',
):
    """
    Fuse historical per-sample age first, then aggregate by class into class-level weights.
    - Supports NaN: unseen samples are excluded from the statistics
    - Finally normalize to mean=1
    """
    if fuse not in ('ema', 'avg'):
        raise ValueError(f"fuse must be 'ema' or 'avg', got {fuse}")

    if len(age_hist_list) == 0:
        return np.ones(n_cls, dtype=np.float64)

    # First fuse the per-sample history to obtain an age_fused vector of length n
    if fuse == 'ema':
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0,1], got {rho}")
        if rho == 0.0:
            return np.ones(n_cls, dtype=np.float64)

        age_fused = np.asarray(age_hist_list[0], dtype=np.float64).copy()
        for t in range(1, len(age_hist_list)):
            age_t = np.asarray(age_hist_list[t], dtype=np.float64)

            prev_fin = np.isfinite(age_fused)
            curr_fin = np.isfinite(age_t)

            out = age_fused.copy()
            both = prev_fin & curr_fin
            out[both] = (1.0 - rho) * age_fused[both] + rho * age_t[both]

            only_curr = (~prev_fin) & curr_fin
            out[only_curr] = age_t[only_curr]

            age_fused = out

    else:
        age_stack = np.stack(
            [np.asarray(a, dtype=np.float64) for a in age_hist_list],
            axis=0
        )
        age_fused = np.nanmean(age_stack, axis=0)

    y_all = np.asarray(y_all, dtype=np.int64)

    avg_age_per_class = np.full(n_cls, np.nan, dtype=np.float64)
    for cls_id in range(n_cls):
        m = (y_all == cls_id) & np.isfinite(age_fused)
        if m.any():
            avg_age_per_class[cls_id] = age_fused[m].mean()

    finite_cls = np.isfinite(avg_age_per_class)
    if not finite_cls.any():
        return np.ones(n_cls, dtype=np.float64)

    mean_w = float(np.mean(avg_age_per_class[finite_cls]))
    if (not np.isfinite(mean_w)) or (mean_w <= 0.0):
        return np.ones(n_cls, dtype=np.float64)

    w = np.ones(n_cls, dtype=np.float64)
    w[finite_cls] = avg_age_per_class[finite_cls] / mean_w
    return w.astype(np.float64)


def compute_sample_weights_from_age_history(
        age_hist_list,
        n_samples: int,
        rho: float = 0.5,
        fuse: str = 'ema',
):
    """
    Fuse historical per-sample age into sample-level weights.
    - Supports NaN: unseen samples are excluded from fusion
    - Finally normalize to mean=1
    """
    if fuse not in ('ema', 'avg'):
        raise ValueError(f"fuse must be 'ema' or 'avg', got {fuse}")

    if len(age_hist_list) == 0:
        return np.ones(n_samples, dtype=np.float64)

    if fuse == 'ema':
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0,1], got {rho}")
        if rho == 0.0:
            return np.ones(n_samples, dtype=np.float64)

        age_fused = np.asarray(age_hist_list[0], dtype=np.float64).copy()
        for t in range(1, len(age_hist_list)):
            age_t = np.asarray(age_hist_list[t], dtype=np.float64)

            prev_fin = np.isfinite(age_fused)
            curr_fin = np.isfinite(age_t)

            out = age_fused.copy()
            both = prev_fin & curr_fin
            out[both] = (1.0 - rho) * age_fused[both] + rho * age_t[both]

            only_curr = (~prev_fin) & curr_fin
            out[only_curr] = age_t[only_curr]

            age_fused = out

    else:
        age_stack = np.stack(
            [np.asarray(a, dtype=np.float64) for a in age_hist_list],
            axis=0
        )
        age_fused = np.nanmean(age_stack, axis=0)

    if age_fused.shape[0] != n_samples:
        raise ValueError(
            f"age_fused length mismatch: got {age_fused.shape[0]}, expected {n_samples}"
        )

    finite = np.isfinite(age_fused)
    if not finite.any():
        return np.ones(n_samples, dtype=np.float64)

    mean_w = float(np.mean(age_fused[finite]))
    if (not np.isfinite(mean_w)) or (mean_w <= 0.0):
        return np.ones(n_samples, dtype=np.float64)

    w = np.ones(n_samples, dtype=np.float64)
    w[finite] = age_fused[finite] / mean_w
    return w.astype(np.float64)


# new for loss 11
# new: CB (Class-Balanced) class weights
def compute_cb_class_weights(
        class_counts: np.ndarray,
        beta: float = 0.9,
        eps: float = 1e-12,
) -> np.ndarray:
    """
    Class-Balanced (CB) weight:
        w_c = (1 - beta) / (1 - beta^{n_c})
    Then normalize to mean=1.

    Args:
        class_counts: shape (C,), each entry n_c > 0
        beta: float in [0,1)
        eps: avoid division by zero
    Returns:
        weights: shape (C,), float64, mean=1
    """
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0,1), got {beta}")

    counts = np.asarray(class_counts, dtype=np.float64)
    # avoid zero count (shouldn't happen for CIFAR100-LT, but keep safe)
    counts = np.clip(counts, a_min=1.0, a_max=None)

    effective_num = 1.0 - np.power(beta, counts)  # (C,)
    weights = (1.0 - beta) / np.clip(effective_num, a_min=eps, a_max=None)

    # normalize weights to mean=1
    num_classes = len(weights)
    weights = weights / np.sum(weights) * num_classes
    return weights.astype(np.float64)


# =======================
# TF-official ResNetCifar(v1) -> PyTorch implementation
# Matches: cifar_model.ResNetCifar + model_base.ResNet (basic block, Plan-A shortcut)
# =======================

class BasicBlockCifarV1(nn.Module):
    """
    ResNet V1 basic block (post-activation):
      conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> add(shortcut) -> ReLU

    Shortcut for (stride>1 or channel change): Plan-A
      avg_pool(stride) + zero-pad channels to match out_planes
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride,
                 bn_decay=0.9, bn_eps=1e-5):
        super().__init__()
        # TF batch_norm decay -> PyTorch momentum:
        # TF: moving = decay*moving + (1-decay)*batch
        # PT: running = (1-momentum)*running + momentum*batch
        # so momentum = 1 - decay
        bn_momentum = 1.0 - bn_decay

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=bn_eps, momentum=bn_momentum)

        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

    def _shortcut_plan_a(self, x):
        # If stride > 1: downsample spatially by avg_pool with SAME padding in TF.
        # PyTorch avg_pool2d with kernel=stride, stride=stride matches shape for CIFAR sizes.
        if self.stride > 1:
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, ceil_mode=False)

        # If channels mismatch: pad zeros equally on both sides of channel dimension
        if self.in_planes != self.out_planes:
            ch = self.out_planes - self.in_planes
            pad1 = ch // 2
            pad2 = ch - pad1
            zeros1 = torch.zeros(x.size(0), pad1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            zeros2 = torch.zeros(x.size(0), pad2, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat([zeros1, x, zeros2], dim=1)
        return x

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = x
        if (self.stride != 1) or (self.in_planes != self.out_planes):
            shortcut = self._shortcut_plan_a(shortcut)

        out = out + shortcut
        out = F.relu(out, inplace=True)
        return out


class ResNetCifarTorch(nn.Module):
    """
    Matches TF cifar_model.ResNetCifar for version='v1' (basic block):
      - n = (num_layers - 2) // 6
      - filters = [16, 16, 32, 64]
      - strides = [1, 2, 2]
      - stem: conv3x3(16) + BN + ReLU
      - stages: 3 stages, each has n blocks
      - head: global_avg_pool + fully_connected (dense)
    """

    def __init__(self, num_layers=32, num_classes=100,
                 bn_decay=0.9, bn_eps=1e-5,
                 loss_type='softmax'):
        super().__init__()
        assert (num_layers - 2) % 6 == 0, "For CIFAR ResNet v1, num_layers must satisfy (L-2)%6==0"
        self.n = (num_layers - 2) // 6  # TF: self.n :contentReference[oaicite:10]{index=10}
        self.num_classes = num_classes
        self.loss_type = loss_type

        bn_momentum = 1.0 - bn_decay

        # TF stem: conv(3,16,1) + BN + ReLU :contentReference[oaicite:11]{index=11}
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=bn_eps, momentum=bn_momentum)

        # TF: filters=[16,16,32,64], strides=[1,2,2] :contentReference[oaicite:12]{index=12}
        self.stage1 = self._make_stage(16, 16, blocks=self.n, first_stride=1,
                                       bn_decay=bn_decay, bn_eps=bn_eps)
        self.stage2 = self._make_stage(16, 32, blocks=self.n, first_stride=2,
                                       bn_decay=bn_decay, bn_eps=bn_eps)
        self.stage3 = self._make_stage(32, 64, blocks=self.n, first_stride=2,
                                       bn_decay=bn_decay, bn_eps=bn_eps)

        # TF head: global_avg_pool -> dense(num_classes) :contentReference[oaicite:13]{index=13}
        self.linear = nn.Linear(64, num_classes)

        # TF: if loss_type is sigmoid/focal, dense bias init = -log(out_dim - 1) :contentReference[oaicite:14]{index=14}
        # You already do bias fill in main(); we do not force-overwrite it here, but the default initialization is kept consistent with TF.
        if self.loss_type in ['sigmoid', 'focal']:
            with torch.no_grad():
                self.linear.bias.fill_(-math.log(num_classes - 1))

    @staticmethod
    def _make_stage(in_planes, out_planes, blocks, first_stride, bn_decay, bn_eps):
        layers = (
                [BasicBlockCifarV1(in_planes, out_planes, stride=first_stride,
                                   bn_decay=bn_decay, bn_eps=bn_eps)]
                + [BasicBlockCifarV1(out_planes, out_planes, stride=1,
                                     bn_decay=bn_decay, bn_eps=bn_eps)
                   for _ in range(1, blocks)]
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Note: TF applies x=x/128-1 in forward_pass :contentReference[oaicite:16]{index=16}
        # Your current transforms already apply an equivalent normalization, so we do not repeat it here to avoid double normalization.

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        # TF global_avg_pool: reduce_mean over H,W :contentReference[oaicite:17]{index=17}
        out = out.mean(dim=(2, 3))
        out = self.linear(out)
        return out


def set_classifier_bias(model, bias_value: float):
    """
    Compatible with:
    - ResNetCifarTorch -> model.linear
    - torchvision VisionTransformer -> model.heads.head
    - some other naming conventions -> model.head
    """
    with torch.no_grad():
        if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
            if model.linear.bias is None:
                raise RuntimeError("ResNet classifier bias is None")
            model.linear.bias.fill_(bias_value)
            return

        if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
            if model.heads.head.bias is None:
                raise RuntimeError("ViT classifier bias is None")
            model.heads.head.bias.fill_(bias_value)
            return

        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            if model.head.bias is None:
                raise RuntimeError("Classifier bias is None")
            model.head.bias.fill_(bias_value)
            return

        raise AttributeError(
            "Cannot find classifier bias layer: expected model.linear, model.heads.head, or model.head")


def build_model(args, num_classes):
    if args.model == 'resnet32':
        return ResNetCifarTorch(
            num_layers=32,
            num_classes=num_classes,
            bn_decay=0.9,
            bn_eps=1e-5,
            loss_type=args.loss_type
        )

    if args.model == 'resnet34':
        return torchvision.models.resnet34(
            weights=None,
            num_classes=num_classes,
        )

    if args.model == 'cifar_vit':
        return VisionTransformer(
            image_size=32,
            patch_size=args.vit_patch_size,
            num_layers=args.vit_depth,
            num_heads=args.vit_num_heads,
            hidden_dim=args.vit_embed_dim,
            mlp_dim=args.vit_mlp_dim,
            num_classes=num_classes,
            dropout=args.vit_dropout,
            attention_dropout=args.vit_dropout,
        )

    raise ValueError(f"Unknown model type: {args.model}")


def build_optimizer(model, args):
    """
    torchvision ViT: use AdamW, and do not apply weight decay to bias / norm / pos_embedding / class_token
    """
    if args.model == 'resnet32':
        wd = args.weight_decay

        if args.loss_type in ['sigmoid', 'focal']:
            params_decay, params_nodecay = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name == 'linear.bias':
                    params_nodecay.append(p)
                else:
                    params_decay.append(p)

            opt = torch.optim.SGD(
                [{'params': params_decay, 'weight_decay': wd},
                 {'params': params_nodecay, 'weight_decay': 0.0}],
                lr=args.lr,
                momentum=args.momentum
            )
        else:
            opt = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=wd
            )

        return opt, args.lr

    if args.model == 'resnet34':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        return opt, args.lr

    if args.model == 'cifar_vit':
        decay, no_decay = [], []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            name_lower = name.lower()
            if (
                    name.endswith('.bias')
                    or 'pos_embedding' in name_lower
                    or 'class_token' in name_lower
                    or '.ln_' in name_lower
                    or 'norm' in name_lower
            ):
                no_decay.append(p)
            else:
                decay.append(p)

        opt = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': args.vit_weight_decay},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=args.vit_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return opt, args.vit_lr

    raise ValueError(f"Unknown model type: {args.model}")


class _ImbalancedCIFARMixin:
    """
    Shared: generate CIFAR long-tail / imbalanced training sets
    Assumes the subclass is torchvision.datasets.CIFAR10 or CIFAR100, and that self.data / self.targets exist
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_img_num_per_cls(self, num_classes, imb_type, imb_factor):
        img_max = len(self.data) / num_classes

        img_num_per_cls = []
        if imb_type == "exp":
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
            new_targets.extend([int(cls)] * int(keep_num))

        new_data = np.concatenate(new_data, axis=0)
        self.data = new_data
        self.targets = new_targets


class ImbalancedCIFAR10(_ImbalancedCIFARMixin, torchvision.datasets.CIFAR10):
    def __init__(self, root, imb_type="exp", imb_factor=100, rand_seed=42,
                 train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        if train:
            img_num_per_cls = self._get_img_num_per_cls(
                num_classes=10, imb_type=imb_type, imb_factor=imb_factor
            )
            self._gen_imbalanced_data(img_num_per_cls, rand_seed)


class ImbalancedCIFAR100(_ImbalancedCIFARMixin, torchvision.datasets.CIFAR100):
    def __init__(self, root, imb_type="exp", imb_factor=100, rand_seed=42,
                 train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        if train:
            img_num_per_cls = self._get_img_num_per_cls(
                num_classes=100, imb_type=imb_type, imb_factor=imb_factor
            )
            self._gen_imbalanced_data(img_num_per_cls, rand_seed)


def adjust_learning_rate(optimizer, epoch, args, base_lr=None):
    """Schedule of learning rate decay"""
    if base_lr is None:
        base_lr = args.lr

    epoch = epoch + 1
    if epoch >= 180:
        lr = base_lr * 0.0001
    elif epoch >= 160:
        lr = base_lr * 0.01
    else:
        lr = base_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    args = parse_args()

    if args.setting in ['online', 'online_dynamic_lt']:
        if args.dataset in ['imagenet', 'places365'] and args.model == 'resnet34':
            args.batch = 256
            args.eval_batch = 256
            args.epochs = 200
            args.lr = 0.01
            args.momentum = 0.9
            args.weight_decay = 1e-4
            args.workers = 16
        else:
            args.batch = 128
            args.epochs = 200
            args.lr = 0.1
            args.vit_lr = 0.1

    if args.setting in ['online', 'online_dynamic_lt'] and args.weight_scope != 'class':
        raise ValueError(
            "Online-like settings only support class-level age weighting, so --weight_scope must be 'class'.")

    if args.setting in ['online', 'online_dynamic_lt'] and args.sampler_weight_type != 'none':
        raise ValueError("Online-like settings do NOT support resampling (sampler_weight_type must be 'none').")

    if args.setting == 'online_dynamic_lt':
        if args.buffer_mode != 'fixed':
            raise ValueError("online_dynamic_lt currently only supports --buffer_mode fixed")

    dataset_num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'places365': 365,
        'imagenet_subset': 1000,
        'places365_subset': 365,
    }
    num_classes = dataset_num_classes[args.dataset]

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    rho_tag = f"{args.age_rho:g}".replace(".", "p")

    base_name = (
        f"{args.dataset}_{args.model}_if{int(args.imb_factor)}_"
        f"{args.setting}_{args.loss_type}_{args.weight_scope}_"
        f"{args.loss_weight_type}_{args.sampler_weight_type}_"
        f"rho{rho_tag}_seed{args.seed}_period{args.lt_shuffle_period}_"
        f"ep{args.epochs}"
    )

    run_name = f"{ts}_{base_name}"
    logger = init_logger(run_name)

    args_line = ", ".join(f"{k}={v}" for k, v in vars(args).items())
    logger.info(f"Args: {args_line}")

    # ===== buffer sanity check =====
    if args.setting in ['online', 'online_dynamic_lt']:
        if args.buffer_mode == 'fixed':
            assert args.buffer_size > 0, "buffer_size must be > 0 when using fixed buffer"
        elif args.buffer_mode == 'infinite':
            # Only warn if the user explicitly sets buffer_size
            logger.warning(
                f"buffer_mode='infinite': buffer_size={args.buffer_size} is ignored"
            )

    # ===== (Added) Prepare for the log footer: parameter string + month-day-hour-minute-second =====
    args_kv_line = ", ".join(f"{k}={v}" for k, v in vars(args).items())
    ts_md_hms = "-".join(ts.split("-")[1:])  # "YYYY-MM-DD-HH-MM-SS" -> "MM-DD-HH-MM-SS"

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
        transforms.Lambda(lambda t: t * (255.0 / 128.0) - 1.0),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * (255.0 / 128.0) - 1.0),
    ])

    imagenet_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    imagenet_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if args.dataset in ['cifar10', 'cifar100']:
        imb_train_dataset = ImbalancedCIFAR10 if args.dataset == 'cifar10' else ImbalancedCIFAR100
        test_dataset = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100

        source_imb_factor = 1 if args.setting == 'online_dynamic_lt' else args.imb_factor

        # Training set: long-tail version (with random augmentation)
        train_set = imb_train_dataset(
            root='./data',
            train=True,
            download=True,
            transform=transform_train,
            imb_type="exp",  # "exp" is the most commonly used option; "step" is also supported
            imb_factor=source_imb_factor,  # Typical values: 200/100/50/10
            rand_seed=args.seed
        )

        # Initialize with a default value to satisfy static checkers
        train_class_count = np.zeros(num_classes, dtype=np.int64)

        if isinstance(train_set, (ImbalancedCIFAR10, ImbalancedCIFAR100)):
            cls_counts = np.bincount(np.array(train_set.targets), minlength=num_classes)
            logger.info(f"Imbalanced train class counts: {cls_counts.tolist()}")

            # Store the training-set per-class sample counts for the later per-class CSV
            train_class_count = cls_counts.astype(np.int64)

            # logger.info("Class index -> sample count:")
            # for c, n in enumerate(cls_counts.tolist()):
            #     logger.info(f"class{c:02d}: {n}")

        # Keep the test set balanced as in the original dataset
        test_set = test_dataset(
            root='./data',
            train=False,
            download=True,
            transform=transform_test
        )

        train_set = IndexedDataset(train_set)

        # Store the test-set per-class sample counts
        test_targets = np.array(test_set.targets, dtype=np.int64)
        test_class_count = np.bincount(test_targets, minlength=num_classes)

        # For training-set evaluation: use the same long-tail training set without random augmentation
        train_eval_set = imb_train_dataset(
            root='./data',
            train=True,
            download=False,
            transform=transform_test,
            imb_type="exp",
            imb_factor=source_imb_factor,
            rand_seed=args.seed
        )

        train_eval_set = IndexedDataset(train_eval_set)

    elif args.dataset == 'imagenet':
        train_lmdb_path = os.path.join(args.data_root, 'train.lmdb')
        train_targets_path = os.path.join(args.data_root, 'train_targets.npy')
        val_lmdb_path = os.path.join(args.data_root, 'val.lmdb')
        val_targets_path = os.path.join(args.data_root, 'val_targets.npy')
        for path in [
            train_lmdb_path,
            train_targets_path,
            val_lmdb_path,
            val_targets_path,
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"ImageNet file not found: {path}")
        train_base_set = LMDBImageDataset(
            lmdb_path=train_lmdb_path,
            targets_path=train_targets_path,
            transform=imagenet_transform_train,
        )
        train_eval_base_set = LMDBImageDataset(
            lmdb_path=train_lmdb_path,
            targets_path=train_targets_path,
            transform=imagenet_transform_test,
        )
        test_set = LMDBImageDataset(
            lmdb_path=val_lmdb_path,
            targets_path=val_targets_path,
            transform=imagenet_transform_test,
        )
        train_targets = np.asarray(
            train_base_set.targets,
            dtype=np.int64
        )
        test_targets = np.asarray(
            test_set.targets,
            dtype=np.int64
        )
        if len(train_targets) != 1281167:
            raise ValueError(
                f"ImageNet train size mismatch: "
                f"{len(train_targets)} != 1281167"
            )
        if len(test_targets) != 50000:
            raise ValueError(
                f"ImageNet val size mismatch: "
                f"{len(test_targets)} != 50000"
            )
        if len(np.unique(train_targets)) != 1000:
            raise ValueError("ImageNet train must contain 1000 classes")
        if len(np.unique(test_targets)) != 1000:
            raise ValueError("ImageNet val must contain 1000 classes")
        train_class_count = np.bincount(
            train_targets,
            minlength=num_classes,
        ).astype(np.int64)
        test_class_count = np.bincount(
            test_targets,
            minlength=num_classes,
        ).astype(np.int64)
        logger.info(
            f"Full ImageNet train samples={len(train_targets)}, "
            f"val samples={len(test_targets)}, "
            f"classes={num_classes}"
        )
        logger.info(
            f"ImageNet train class count min/max="
            f"{train_class_count.min()}/"
            f"{train_class_count.max()}"
        )
        train_set = IndexedDataset(train_base_set)
        train_eval_set = IndexedDataset(train_eval_base_set)

    elif args.dataset == 'places365':
        train_lmdb_path = os.path.join(args.data_root, 'train.lmdb')
        train_targets_path = os.path.join(args.data_root, 'train_targets.npy')
        val_lmdb_path = os.path.join(args.data_root, 'val.lmdb')
        val_targets_path = os.path.join(args.data_root, 'val_targets.npy')

        for path in [
            train_lmdb_path,
            train_targets_path,
            val_lmdb_path,
            val_targets_path,
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Places365 file not found: {path}"
                )

        train_base_set = LMDBImageDataset(
            lmdb_path=train_lmdb_path,
            targets_path=train_targets_path,
            transform=imagenet_transform_train,
        )

        train_eval_base_set = LMDBImageDataset(
            lmdb_path=train_lmdb_path,
            targets_path=train_targets_path,
            transform=imagenet_transform_test,
        )

        test_set = LMDBImageDataset(
            lmdb_path=val_lmdb_path,
            targets_path=val_targets_path,
            transform=imagenet_transform_test,
        )

        train_targets = np.asarray(
            train_base_set.targets,
            dtype=np.int64
        )

        test_targets = np.asarray(
            test_set.targets,
            dtype=np.int64
        )

        if len(train_targets) != 1803460:
            raise ValueError(
                f"Places365 train size mismatch: "
                f"{len(train_targets)} != 1803460"
            )

        if len(test_targets) != 36500:
            raise ValueError(
                f"Places365 val size mismatch: "
                f"{len(test_targets)} != 36500"
            )

        if len(np.unique(train_targets)) != 365:
            raise ValueError(
                "Places365 train must contain 365 classes"
            )

        if len(np.unique(test_targets)) != 365:
            raise ValueError(
                "Places365 val must contain 365 classes"
            )

        train_class_count = np.bincount(
            train_targets,
            minlength=num_classes,
        ).astype(np.int64)

        test_class_count = np.bincount(
            test_targets,
            minlength=num_classes,
        ).astype(np.int64)

        logger.info(
            f"Full Places365 train samples={len(train_targets)}, "
            f"val samples={len(test_targets)}, "
            f"classes={num_classes}"
        )

        logger.info(
            f"Places365 train class count min/max="
            f"{train_class_count.min()}/"
            f"{train_class_count.max()}"
        )

        train_set = IndexedDataset(train_base_set)
        train_eval_set = IndexedDataset(train_eval_base_set)

    else:
        subset_name = (
            'imagenet_subset'
            if args.dataset == 'imagenet_subset'
            else 'places365_subset'
        )

        subset_root = os.path.join(args.data_root, subset_name)
        train_root = os.path.join(subset_root, 'train')
        val_root = os.path.join(subset_root, 'val')

        if not os.path.isdir(train_root):
            raise FileNotFoundError(
                f"Training directory not found: {train_root}"
            )
        if not os.path.isdir(val_root):
            raise FileNotFoundError(
                f"Validation directory not found: {val_root}"
            )

        train_base_set = torchvision.datasets.ImageFolder(
            root=train_root,
            transform=transform_train
        )
        train_eval_base_set = torchvision.datasets.ImageFolder(
            root=train_root,
            transform=transform_test
        )
        test_set = torchvision.datasets.ImageFolder(
            root=val_root,
            transform=transform_test
        )

        if len(train_base_set.classes) != num_classes:
            raise ValueError(
                f"{args.dataset}: found {len(train_base_set.classes)} "
                f"training classes, expected {num_classes}"
            )
        if len(test_set.classes) != num_classes:
            raise ValueError(
                f"{args.dataset}: found {len(test_set.classes)} "
                f"validation classes, expected {num_classes}"
            )
        if train_base_set.class_to_idx != train_eval_base_set.class_to_idx:
            raise ValueError(
                "train and train_eval class_to_idx mismatch"
            )
        if train_base_set.class_to_idx != test_set.class_to_idx:
            raise ValueError(
                "train and validation class_to_idx mismatch"
            )

        train_class_count = np.bincount(
            np.asarray(train_base_set.targets, dtype=np.int64),
            minlength=num_classes
        ).astype(np.int64)
        logger.info(
            f"Subset train class counts: {train_class_count.tolist()}"
        )

        test_targets = np.array(test_set.targets, dtype=np.int64)
        test_class_count = np.bincount(
            test_targets,
            minlength=num_classes
        )

        train_set = IndexedDataset(train_base_set)
        train_eval_set = IndexedDataset(train_eval_base_set)

    # ======= (Added) idx alignment sanity check: crucial! =======
    train_labels_for_idx = np.asarray(train_set.base.targets, dtype=np.int64)
    eval_labels_for_idx = np.asarray(train_eval_set.base.targets, dtype=np.int64)

    assert len(train_labels_for_idx) == len(eval_labels_for_idx), \
        f"targets length mismatch: train={len(train_labels_for_idx)} eval={len(eval_labels_for_idx)}"

    assert np.array_equal(train_labels_for_idx, eval_labels_for_idx), \
        ("train_set.base.targets != train_eval_set.base.targets (index order mismatch). "
         "This will misalign idx-based age/cb sample weights with the training sampler/dataloader.")

    # new for loss 2
    # ======= (Added) labels for class aggregation by idx (aligned with train_eval_set idx) =======
    train_eval_labels = np.array(train_eval_set.base.targets, dtype=np.int64)  # shape=(n,)

    # Precompute class indices only for full ImageNet.
    # Other datasets keep the original behavior.
    dynamic_class_indices = None

    if args.dataset in ['imagenet', 'places365']:
        dynamic_class_indices = [
            np.where(train_eval_labels == c)[0]
            for c in range(num_classes)
        ]

    g = torch.Generator()
    g.manual_seed(args.seed)

    def _worker_init(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, generator=g,
    #                                            num_workers=args.workers, pin_memory=True, worker_init_fn=_worker_init)
    # ======= (Added) sanity check: sample-level age / sampler both depend on idx alignment =======
    assert len(train_set) == len(train_eval_set), \
        f"train_set(len={len(train_set)}) must equal train_eval_set(len={len(train_eval_set)}) for age-based sampling"
    # # ======= (Added) Optional: use WeightedRandomSampler for age-based resampling (replacement=True) =======
    # train_sampler_weights = None
    # logger.info(f"SAMPLER_WEIGHT_TYPE = {args.sampler_weight_type}")
    # if args.sampler_weight_type != 'none':
    #     # Initial weights are all 1; w_age_s will be updated in the loop every epoch
    #     train_sampler_weights = torch.ones(len(train_set), dtype=torch.double)  # CPU double
    #     train_sampler = torch.utils.data.WeightedRandomSampler(
    #         weights=train_sampler_weights,
    #         num_samples=len(train_set),  # Fairness: total number of samples is fixed to the training-set size
    #         replacement=True,  # Allow sampling with replacement
    #         generator=g  # Keep it reproducible
    #     )
    #     train_loader = torch.utils.data.DataLoader(
    #         train_set,
    #         batch_size=args.batch,
    #         shuffle=False,  # sampler and shuffle are mutually exclusive
    #         sampler=train_sampler,
    #         generator=g,  # Keep worker seed reproducible
    #         num_workers=args.workers,
    #         pin_memory=True,
    #         worker_init_fn=_worker_init
    #     )
    # else:
    #     # Original baseline: shuffle=True as in your code
    #     train_loader = torch.utils.data.DataLoader(
    #         train_set,
    #         batch_size=args.batch,
    #         shuffle=True,
    #         generator=g,
    #         num_workers=args.workers,
    #         pin_memory=True,
    #         worker_init_fn=_worker_init
    #     )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch, shuffle=False, generator=g,
                                              num_workers=args.workers, pin_memory=True, worker_init_fn=_worker_init)

    # new
    train_eval_loader = torch.utils.data.DataLoader(train_eval_set, batch_size=args.eval_batch, shuffle=False,
                                                    num_workers=args.workers, pin_memory=True,
                                                    worker_init_fn=_worker_init)

    # ======= (Added) Test-set Age evaluation: test_set + idx, and no shuffle =======
    test_eval_set = IndexedDataset(test_set)
    test_eval_loader = torch.utils.data.DataLoader(
        test_eval_set,
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=_worker_init
    )

    # model = ResNet34_CIFAR(num_classes=100).to(device)
    # model = resnet32(num_classes=100, use_norm=False).to(device)
    # model = ResNetCifarTorch(
    #     num_layers=32,  # You use resnet32 -> in TF this corresponds to num_layers=32
    #     num_classes=num_classes,
    #     bn_decay=0.9,  # Match the TF batch_norm_decay semantics (commonly 0.9); momentum will be 1-decay
    #     bn_eps=1e-5,  # Match the common TF batch_norm_epsilon setting
    #     loss_type=args.loss_type
    # ).to(device)
    #
    # if args.loss_type in ['sigmoid', 'focal']:
    #     num_classes_local = num_classes
    #     pi = 1.0 / num_classes_local
    #     b0 = -math.log((1.0 - pi) / pi)
    #     with torch.no_grad():
    #         model.linear.bias.fill_(b0)
    model = build_model(args, num_classes).to(device)

    if args.loss_type in ['sigmoid', 'focal']:
        pi = 1.0 / num_classes
        b0 = -math.log((1.0 - pi) / pi)
        set_classifier_bias(model, b0)

    # new for loss 3
    # loss_fn = nn.CrossEntropyLoss()  # (deleted/modified) no longer fixed

    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # wd = args.weight_decay
    #
    # if args.loss_type in ['sigmoid', 'focal']:
    #     params_decay, params_nodecay = [], []
    #     for name, p in model.named_parameters():
    #         if not p.requires_grad:
    #             continue
    #         if name == 'linear.bias':
    #             params_nodecay.append(p)
    #         else:
    #             params_decay.append(p)
    #     opt = torch.optim.SGD(
    #         [{'params': params_decay, 'weight_decay': wd},
    #          {'params': params_nodecay, 'weight_decay': 0.0}],
    #         lr=args.lr, momentum=args.momentum
    #     )
    # else:
    #     opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=wd)
    opt, base_lr = build_optimizer(model, args)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)

    # ===== LR schedule: warmup(5) + step decay at 160/180 (paper setting) =====
    # warmup_epochs = 5

    n = len(train_eval_set)
    epoch_val_loader = None
    online_val_indices = None
    online_class_age = None
    online_class_age_history = None
    online_class_age_ema = None
    current_lt_rank_perm = None
    epoch_access_loader = None
    access_eval_loader = None
    prev_seen_mask = None
    current_mask = None
    total_seen = 0  # Record how many samples have been seen so far (N)

    if args.setting == 'offline':
        if args.epoch_drop_percent > 0.0:
            age_curr = np.full(n, np.nan, dtype=np.float64)
        else:
            age_curr = np.ones(n, dtype=np.float64)
        seen_mask = np.zeros(n, dtype=bool)
        age_history = []
    else:
        age_curr = None
        seen_mask = None
        age_history = []  # Keep the variable name to avoid later reference errors; online does not use it for training age
        online_class_age = np.ones(num_classes, dtype=np.float64)
        online_class_age_history = []
        online_class_age_ema = None
        online_val_indices = []  # crucial: allow duplicate appends

    n_test = len(test_eval_set)
    age_curr_test = np.ones(n_test, dtype=np.float64)
    age_history_test = []

    best_acc = -1.0
    best_path = None  # Give a placeholder to satisfy static checkers
    test_acc = float("nan")

    # --- CB weight (fixed) ---
    w_cb = compute_cb_class_weights(train_class_count, beta=args.cb_beta)  # (C,), mean=1
    # print(f"w_cb = {w_cb}")

    # --- (Added) CB-RS: map class-level CB weights to per-sample weights (aligned with idx) ---
    w_cb_s = w_cb[train_eval_labels]  # shape (n,), sample 'i' uses class weight of its label
    # print(f"w_cb_s.mean = {w_cb_s.mean()}")
    # print(f"len(w_cb_s) = {len(w_cb_s)}")
    # print("w_cb_s[-100:] =", w_cb_s[-100:])

    # Number of samples retained at the end of each epoch
    effective_keep_num = len(train_set)
    if args.epoch_drop_percent > 0.0:
        effective_keep_num = max(
            1,
            int(round(len(train_set) * (100.0 - args.epoch_drop_percent) / 100.0))
        )

    steps_per_epoch = math.ceil(effective_keep_num / args.batch)
    if args.setting == 'offline':
        warmup_epochs = 5
        warmup_steps = warmup_epochs * steps_per_epoch
    else:
        # warmup_epochs = 0
        warmup_steps = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):

        # ====== Added: at the start of each epoch or every few epochs, randomly drop the specified percentage of training samples ======
        if args.setting == 'online_dynamic_lt':
            if args.lt_shuffle_period < 1:
                raise ValueError(f"--lt_shuffle_period must be >= 1, got {args.lt_shuffle_period}")

            # Reshuffle the head/tail class order only every K epochs
            # Initialize once at epoch=1
            need_refresh_lt_order = (
                    current_lt_rank_perm is None
                    or ((epoch - 1) % args.lt_shuffle_period == 0)
            )

            if need_refresh_lt_order:
                current_lt_rank_perm = None  # Let the function generate a new random order internally
            else:
                # Reuse the class order from the previous regime
                pass

            epoch_train_set, epoch_keep_indices, epoch_class_counts, epoch_rank_perm = build_epoch_dynamic_longtail_subset(
                base_dataset=train_set,
                labels=train_eval_labels,
                num_classes=num_classes,
                imb_factor=args.imb_factor,
                # Here args.imb_factor denotes the target IF for the dynamic subset of each epoch
                seed=args.seed,
                epoch=epoch,
                imb_type="exp",
                rank_perm=current_lt_rank_perm,
                class_indices=dynamic_class_indices,
            )

            current_lt_rank_perm = epoch_rank_perm.copy()

            if need_refresh_lt_order:
                logger.info(
                    f"Epoch {epoch}: REFRESH dynamic LT class order "
                    f"(period={args.lt_shuffle_period}), "
                    f"head_class={int(epoch_rank_perm[0])}, tail_class={int(epoch_rank_perm[-1])}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: REUSE dynamic LT class order "
                    f"(period={args.lt_shuffle_period}), "
                    f"head_class={int(epoch_rank_perm[0])}, tail_class={int(epoch_rank_perm[-1])}"
                )

            logger.info(
                f"Epoch {epoch}: dynamic long-tail subset built from balanced source, "
                f"target_if={args.imb_factor}, keep_num={len(epoch_train_set)}/{len(train_set)}"
            )
        else:
            epoch_train_set, epoch_keep_indices = build_epoch_train_subset(
                train_set,
                drop_percent=args.epoch_drop_percent,
                seed=args.seed,
                epoch=epoch
            )

        if args.setting in ['online', 'online_dynamic_lt']:

            for idx in epoch_keep_indices:
                total_seen += 1  # This is the N-th sample observed so far
                if args.buffer_mode == 'infinite':  # original logic
                    online_val_indices.append(int(idx))
                else:  # fixed buffer（Reservoir Sampling）
                    if len(online_val_indices) < args.buffer_size:  # buffer not yet full
                        online_val_indices.append(int(idx))
                    else:
                        j = np.random.randint(0, total_seen)  # Core probability: K / N
                        if j < args.buffer_size:
                            replace_idx = np.random.randint(0, args.buffer_size)  # replace one item in the buffer
                            online_val_indices[replace_idx] = int(idx)

            epoch_val_set = torch.utils.data.Subset(train_eval_set, online_val_indices)
            epoch_val_loader = torch.utils.data.DataLoader(
                epoch_val_set,
                batch_size=args.eval_batch,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                worker_init_fn=_worker_init
            )

            logger.info(
                f"[Online] validation buffer size={len(online_val_indices)} "
                f"(added {len(epoch_keep_indices)} samples this epoch, duplicates kept)"
            )

        if args.setting == 'offline':
            current_mask = np.zeros(n, dtype=bool)
            current_mask[epoch_keep_indices] = True

            prev_seen_mask = seen_mask.copy()
            seen_mask |= current_mask

            epoch_access_set = torch.utils.data.Subset(train_eval_set, epoch_keep_indices.tolist())
            epoch_access_loader = torch.utils.data.DataLoader(
                epoch_access_set,
                batch_size=args.eval_batch,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                worker_init_fn=_worker_init
            )

            seen_indices = np.flatnonzero(seen_mask).astype(np.int64)
            access_eval_set = torch.utils.data.Subset(train_eval_set, seen_indices.tolist())
            access_eval_loader = torch.utils.data.DataLoader(
                access_eval_set,
                batch_size=args.eval_batch,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                worker_init_fn=_worker_init
            )

        if args.setting == 'online_dynamic_lt':
            logger.info(
                f"Epoch {epoch}: online_dynamic_lt, target_if={args.imb_factor}, "
                f"keep_num={len(epoch_train_set)}/{len(train_set)}"
            )
        else:
            logger.info(
                f"Epoch {epoch}: epoch_drop_percent={args.epoch_drop_percent:.2f}, "
                f"keep_num={len(epoch_train_set)}/{len(train_set)}"
            )

        # ====== 1) First compute the age weights for this epoch ======
        if args.setting == 'offline':
            if epoch == 1:
                w_age_cls = np.ones(num_classes, dtype=np.float64)
                w_age_s = np.ones(n, dtype=np.float64)
            else:
                w_age_cls = compute_class_weights_from_age_history(
                    age_history,
                    train_eval_labels,
                    n_cls=num_classes,
                    rho=args.age_rho,
                    fuse=args.age_fuse
                )
                w_age_s = compute_sample_weights_from_age_history(
                    age_history,
                    n_samples=n,
                    rho=args.age_rho,
                    fuse=args.age_fuse
                )
        else:
            # ===== Correct EMA (recursive version)=====
            if online_class_age_ema is None:
                w_age_cls = online_class_age.copy()
            else:
                w_age_cls = online_class_age_ema.copy()
            # Normalization (very important; otherwise the loss scale will drift)
            mean_w = np.mean(w_age_cls)
            if mean_w > 0:
                w_age_cls = w_age_cls / mean_w
            w_age_s = np.ones(n, dtype=np.float64)

        w_age_cls_s = w_age_cls[train_eval_labels]

        # ====== Added: build train_loader separately for each epoch to ensure samples are dropped before the remaining pipeline ======
        if args.sampler_weight_type != 'none':
            if args.sampler_weight_type == 'age':
                if args.weight_scope == 'sample':
                    sampler_w = w_age_s[epoch_keep_indices]
                    sampler_tag = "AGE-SAMPLE"
                else:
                    sampler_w = w_age_cls_s[epoch_keep_indices]
                    sampler_tag = "AGE-CLS"

            elif args.sampler_weight_type == 'cb':
                sampler_w = w_cb_s[epoch_keep_indices]
                sampler_tag = "CB"

            else:
                raise ValueError(f"Unknown sampler_weight_type: {args.sampler_weight_type}")

            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sampler_w, dtype=torch.double),
                num_samples=len(epoch_train_set),
                replacement=True,
                generator=g
            )

            train_loader = torch.utils.data.DataLoader(
                epoch_train_set,
                batch_size=args.batch,
                shuffle=False,
                sampler=train_sampler,
                generator=g,
                num_workers=args.workers,
                pin_memory=True,
                worker_init_fn=_worker_init
            )

            logger.info(
                f"[Sampler:{sampler_tag}] replacement=True num_samples={len(epoch_train_set)} "
                f"sampler_w(mean/min/max)={sampler_w.mean():.6f}/{sampler_w.min():.6f}/{sampler_w.max():.6f}"
            )

        else:
            train_loader = torch.utils.data.DataLoader(
                epoch_train_set,
                batch_size=args.batch,
                shuffle=True,
                generator=g,
                num_workers=args.workers,
                pin_memory=True,
                worker_init_fn=_worker_init
            )

        # set lr for this epoch (warmup + step decay)
        if args.setting == 'offline':
            if global_step < warmup_steps:
                logger.info(
                    f"Epoch {epoch}: step-warmup active "
                    f"(global_step={global_step}/{warmup_steps}), lr will be set per-step"
                )
            else:
                lr_epoch = adjust_learning_rate(opt, epoch - 1, args, base_lr=base_lr)
                logger.info(f"Epoch {epoch}: lr={lr_epoch:.6e}")
        else:
            # online: fixed lr = 0.1
            lr_epoch = args.lr
            for pg in opt.param_groups:
                pg["lr"] = lr_epoch
            logger.info(f"Epoch {epoch}: online fixed lr={lr_epoch:.6e}")

        # # --- age weight (changes with epoch) ---
        # if epoch == 1:
        #     w_age_cls = np.ones(num_classes, dtype=np.float64)
        #     w_age_s = np.ones(n, dtype=np.float64)  # Added: sample-level (length n, aligned with train_eval_set idx)
        # else:
        #     w_age_cls = compute_class_weights_from_age_history(
        #         age_history,
        #         train_eval_labels,
        #         n_cls=num_classes,
        #         rho=args.age_rho,
        #         fuse=args.age_fuse
        #     )  # (C,)
        #     w_age_s = compute_sample_weights_from_age_history(
        #         age_history,
        #         n_samples=n,
        #         rho=args.age_rho,
        #         fuse=args.age_fuse
        #     )
        #
        # # If sampler uses class-level age, map class weights to each sample
        # w_age_cls_s = w_age_cls[train_eval_labels]
        #
        # # ======= (Added) Use the previous epoch(s) Age -> update the WeightedRandomSampler for this epoch =======
        # # In this way, the sampling distribution of epoch e depends only on historical age_history (i.e., information from epochs < e), consistent with building the sampler from the previous epoch's Age
        # if args.sampler_weight_type != 'none':
        #     if args.sampler_weight_type == 'age':
        #
        #         if args.weight_scope == 'sample':
        #             sampler_w = w_age_s
        #             sampler_tag = "AGE-SAMPLE"
        #         else:
        #             sampler_w = w_age_cls_s
        #             sampler_tag = "AGE-CLS"
        #
        #         # Age-RS: use class-level age weights mapped to each sample
        #         train_sampler_weights.copy_(torch.as_tensor(sampler_w, dtype=torch.double))
        #         logger.info(
        #             f"[Sampler:{sampler_tag}] replacement=True num_samples={len(train_set)} "
        #             f"sampler_w(mean/min/max)={sampler_w.mean():.6f}/{sampler_w.min():.6f}/{sampler_w.max():.6f}"
        #         )
        #
        #     elif args.sampler_weight_type == 'cb':
        #         # CB-RS: use per-sample weights mapped from class-level CB weights
        #         train_sampler_weights.copy_(torch.as_tensor(w_cb_s, dtype=torch.double))
        #         logger.info(
        #             f"[Sampler:CB] replacement=True num_samples={len(train_set)} "
        #             f"w_cb_s(mean/min/max)={w_cb_s.mean():.6f}/{w_cb_s.min():.6f}/{w_cb_s.max():.6f}"
        #         )
        #     else:
        #         raise ValueError(f"Unknown sampler_weight_type: {args.sampler_weight_type}")

        # --- choose final weight according to scope ---
        # loss_type determines the softmax / sigmoid / focal form;
        # loss_weight_type determines the source of loss weights (none / cb / age / cb_age)
        # But you want to switch freely between age only or cb*age, so use loss_weight_type to control the combination
        # ===== weight tensors prepared every epoch =====
        # w_cb_t = torch.tensor(w_cb, dtype=torch.float32, device=device)  # (C,)

        # Re-enable class-level age: losses always use class-level weights
        use_sample_age = (args.weight_scope == 'sample')

        w_final_cls = np.ones(num_classes, dtype=np.float64)
        w_final_s = np.ones(n, dtype=np.float64)

        # ===== choose final weight according to scope =====
        if args.weight_scope == 'class':
            if args.loss_weight_type == 'none':
                w_final_cls = np.ones(num_classes, dtype=np.float64)
            elif args.loss_weight_type == 'cb':
                w_final_cls = w_cb
            elif args.loss_weight_type == 'age':
                w_final_cls = w_age_cls
            elif args.loss_weight_type == 'cb_age':
                w_final_cls = w_cb * w_age_cls
                w_final_cls = w_final_cls / np.mean(w_final_cls)
            else:
                raise ValueError(f"Unknown loss_weight_type: {args.loss_weight_type}")

            w_final_cls_t = torch.tensor(w_final_cls, dtype=torch.float32, device=device)
            w_final_s_t = None

        elif args.weight_scope == 'sample':
            if args.loss_weight_type == 'none':
                w_final_s = np.ones(n, dtype=np.float64)
            elif args.loss_weight_type == 'cb':
                w_final_s = w_cb_s
            elif args.loss_weight_type == 'age':
                w_final_s = w_age_s
            elif args.loss_weight_type == 'cb_age':
                w_final_s = w_cb_s * w_age_s
                w_final_s = w_final_s / np.mean(w_final_s)
            else:
                raise ValueError(f"Unknown loss_weight_type: {args.loss_weight_type}")

            w_final_s_t = torch.tensor(w_final_s, dtype=torch.float32, device=device)
            w_final_cls_t = None

        else:
            raise ValueError(f"Unknown weight_scope: {args.weight_scope}")

        logger.info(
            f"Epoch:[{epoch}/{args.epochs}] "
            f"loss_type={args.loss_type} weight_scope={args.weight_scope} loss_weight_type={args.loss_weight_type} "
            f"sampler_weight_type={args.sampler_weight_type} age_fuse={args.age_fuse} rho={args.age_rho}"
        )

        if args.weight_scope == 'class':
            w_final_log = w_final_cls
        else:
            w_final_log = w_final_s

        logger.info(
            f"weight scope={args.weight_scope} "
            f"w_final(shape={w_final_log.shape}) "
            f"mean/min/max={w_final_log.mean():.6f}/{w_final_log.min():.6f}/{w_final_log.max():.6f}"
        )

        # class_w_t = torch.tensor(class_w, dtype=torch.float32, device=device)
        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.CrossEntropyLoss(weight=class_w_t)

        model.train()
        epoch_start = time.time()
        last_log_time = epoch_start
        running_loss = 0.0
        freq = max(1, len(train_loader) // 10)

        # for step, (x, y) in enumerate(train_loader, start=1):
        for step, (x, y, idx) in enumerate(train_loader, start=1):

            # ---- step-based warmup (linear) ----
            # if global_step < warmup_steps:
            #     lr_now = args.lr * float(global_step + 1) / float(warmup_steps)
            #     for pg in opt.param_groups:
            #         pg["lr"] = lr_now
            if args.setting == 'offline':
                if global_step < warmup_steps:
                    lr_now = base_lr * float(global_step + 1) / float(warmup_steps)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_now
                global_step += 1
            else:
                pass

            # x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)

            logits = model(x)
            # loss = loss_fn(logits, y)
            # if args.loss_type == 'softmax':
            #     # Section 4.1: softmax CE with class weight (CB / age / cb_age)
            #     loss = cb_softmax_ce_loss(logits, y, w_final_t)
            # elif args.loss_type == 'sigmoid':
            #     # Section 4.2: sigmoid CE with weight_bc tiled from w_final
            #     loss = cb_sigmoid_ce_loss(logits, y, w_final_t)
            # elif args.loss_type == 'focal':
            #     # Section 4.3: focal with weight_bc tiled from w_final
            #     loss = cb_focal_loss(logits, y, w_final_t, gamma=args.focal_gamma)
            # else:
            #     raise ValueError(f"Unknown loss_type: {args.loss_type}")

            if not use_sample_age:
                # ===== Keep your original logic: class-level weights (none / cb)=====
                if args.loss_type == 'softmax':
                    loss = cb_softmax_ce_loss(logits, y, w_final_cls_t)
                elif args.loss_type == 'sigmoid':
                    loss = cb_sigmoid_ce_loss(logits, y, w_final_cls_t)
                elif args.loss_type == 'focal':
                    loss = cb_focal_loss(logits, y, w_final_cls_t, gamma=args.focal_gamma)
                else:
                    raise ValueError(f"Unknown loss_type: {args.loss_type}")

            else:
                # ===== Added: sample-level age weights (age / cb_age)=====
                # 1) First take the age weight for each sample (B,)
                w_b = w_final_s_t[idx]  # (B,)

                if step == 1:
                    logger.info(
                        f"[SampleWeight] batch stats: w_b.sum()={w_b.sum().item():.6f}, mean={w_b.mean().item():.4f}, "
                        f"min={w_b.min().item():.4f}, max={w_b.max().item():.4f}"
                    )

                batch_size = y.size(0)

                # 3) Compute per-sample loss, then take the weighted mean
                if args.loss_type == 'softmax':
                    per_sample = F.cross_entropy(logits, y, reduction='none')  # (B,)
                    loss = (w_b * per_sample).sum() / w_b.sum().clamp_min(1e-12)
                    # print(f"w_b.sum()={w_b.sum().clamp_min(1e-12)}")

                elif args.loss_type == 'sigmoid':
                    bsz, num_classes = logits.shape
                    one_hot = F.one_hot(y, num_classes=num_classes).to(dtype=logits.dtype)  # (B,C)
                    bce_bc = F.binary_cross_entropy_with_logits(logits, one_hot, reduction='none')  # (B,C)
                    per_sample = bce_bc.sum(dim=1)  # (B,)
                    loss = (w_b * per_sample).sum() / float(batch_size)

                elif args.loss_type == 'focal':
                    bsz, num_classes = logits.shape
                    one_hot = F.one_hot(y, num_classes=num_classes).to(dtype=logits.dtype)  # (B,C)
                    cross_entropy = F.binary_cross_entropy_with_logits(logits, one_hot, reduction='none')  # (B,C)

                    gamma = args.focal_gamma
                    if gamma == 0.0:
                        modulator = torch.ones_like(logits)
                    else:
                        modulator = torch.exp(-gamma * one_hot * logits - gamma * F.softplus(-logits))  # (B,C)

                    loss_bc = modulator * cross_entropy  # (B,C)
                    per_sample = loss_bc.sum(dim=1)  # (B,)
                    loss = (w_b * per_sample).sum() / float(batch_size)

                else:
                    raise ValueError(f"Unknown loss_type: {args.loss_type}")

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

        # scheduler.step()

        # train_acc = evaluate(model, train_loader, device)
        if args.setting == 'offline':
            if args.epoch_drop_percent > 0.0:
                train_acc = evaluate(model, access_eval_loader, device)
            else:
                train_acc = evaluate(model, train_eval_loader, device)  # changed

        else:
            # online: only used for logging here, not for age
            if args.dataset in ['imagenet', 'places365']:
                train_acc = float("nan")
            else:
                train_acc = evaluate(model, train_eval_loader, device)

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

        # On the training set without augmentation, update Age according to whether the prediction is correct
        if args.setting == 'offline':
            # original logic fully preserved
            if args.epoch_drop_percent > 0.0:
                age_curr = update_age_with_access_constraint(
                    model=model,
                    access_loader=epoch_access_loader,
                    device=device,
                    age_curr=age_curr,
                    prev_seen_mask=prev_seen_mask,
                    current_mask=current_mask,
                )
            else:
                age_curr = update_age_by_correctness(
                    model,
                    train_eval_loader,
                    device,
                    age_curr
                )

            age_history.append(age_curr.copy())

        else:  # online: update class-level age based on the accumulated validation buffer
            # ===== One forward pass to compute accuracy =====
            online_class_acc, online_class_correct, online_class_total = compute_class_stats(
                model=model,
                loader=epoch_val_loader,
                device=device,
                num_classes=num_classes,
            )

            # ===== median threshold =====
            valid_mask = online_class_total > 0
            if valid_mask.any():
                dynamic_threshold = np.median(online_class_acc[valid_mask])
            else:
                dynamic_threshold = 0.0

            # ===== No additional forward pass; update age directly =====
            online_class_age = update_class_age_with_threshold(
                class_age=online_class_age,
                class_acc=online_class_acc,
                class_total=online_class_total,
                acc_threshold=dynamic_threshold,
            )
            logger.info(f"[DynamicThreshold] epoch={epoch} threshold={dynamic_threshold:.4f}")

            online_class_age_history.append(online_class_age.copy())

            # ===== Added: Recursive EMA update (core) =====
            if online_class_age_ema is None:
                online_class_age_ema = online_class_age.copy()
            else:
                online_class_age_ema = (
                        (1.0 - args.age_rho) * online_class_age_ema
                        + args.age_rho * online_class_age
                )

            valid_cls_mask = online_class_total > 0
            if valid_cls_mask.any():
                logger.info(
                    f"[OnlineAge] threshold={dynamic_threshold:.4f} "
                    f"covered_classes={int(valid_cls_mask.sum())}/{num_classes} "
                    f"class_age(mean/min/max)="
                    f"{online_class_age[valid_cls_mask].mean():.6f}/"
                    f"{online_class_age[valid_cls_mask].min():.6f}/"
                    f"{online_class_age[valid_cls_mask].max():.6f}"
                )

        # age_curr = update_age_by_self_information(
        #     model,
        #     train_eval_loader,
        #     device,
        #     age_curr,
        #     loss_type=args.loss_type
        # )

        # Record the Age snapshot for the current epoch (one value per sample)
        # age_history.append(age_curr.copy())

        # ======= (Added) Also update Age once on the test set (same rule: correct -> 1, wrong -> +1) =======
        age_curr_test = update_age_by_correctness(
            model,
            test_eval_loader,  # Note: here we use test_eval_loader (with idx)
            device,
            age_curr_test
        )
        # age_curr_test = update_age_by_self_information(
        #     model,
        #     test_eval_loader,
        #     device,
        #     age_curr_test,
        #     loss_type=args.loss_type
        # )
        age_history_test.append(age_curr_test.copy())

        # age_history: list of length = epochs; each element has shape (n,)
        if args.setting == 'offline':
            _ = np.stack(age_history, axis=0)  # (E, N)  # age_history_np =

        # # === Per-sample printout (age_history, age_history_np as requested)===
        # for i in range(n):
        #     print("====== age history debug (per sample) ======")
        #     print("idx:", i)
        #     print("age_curr:", int(age_curr[i]))  # Age for the current epoch
        #     print("age_history:", [int(v[i]) for v in age_history])  # list of arrays -> age for each epoch
        #     print("age_history_np[:, i]:", age_history_np[:, i].tolist())  # numpy version
        #     print("===========================================")

        # os.makedirs("per_sample_age", exist_ok=True)
        #
        # age_csv_path = os.path.join("per_sample_age", f"{run_name}_per_sample_age.csv")
        # with open(age_csv_path, "w", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     # header:sample_idx, epoch1, epoch2, ...
        #     num_epochs_done = age_history_np.shape[0]
        #     writer.writerow(["sample_idx"] + [f"epoch{e:03d}" for e in range(1, num_epochs_done + 1)])
        #
        #     # one row per sample's age list
        #     for i in range(n):
        #         writer.writerow([i] + age_history_np[:, i].tolist())
        #
        # logger.info(f"Per-sample Age CSV saved to {age_csv_path}")

        # ======= (Added) Save the per-sample Age CSV for the test set =======
        # age_history_test_np = np.stack(age_history_test, axis=0)  # (E, N_test)
        #
        # os.makedirs("per_sample_age_test", exist_ok=True)
        # age_csv_path_test = os.path.join("per_sample_age_test", f"{run_name}_per_sample_age_test.csv")
        #
        # with open(age_csv_path_test, "w", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     num_epochs_done = age_history_test_np.shape[0]
        #     writer.writerow(["sample_idx"] + [f"epoch{e:03d}" for e in range(1, num_epochs_done + 1)])
        #
        #     for i in range(n_test):
        #         writer.writerow([i] + age_history_test_np[:, i].tolist())
        #
        # logger.info(f"Per-sample TEST Age CSV saved to {age_csv_path_test}")

    # ======= After training: save the model from the last epoch =======
    last_path = f'weights/{run_name}_last.pth'
    torch.save(model.state_dict(), last_path)
    logger.info(f"Last epoch model saved to {last_path}")

    # ======= After training: compute per-sample and per-class average Age from age_history =======
    if args.setting == 'offline':
        # age_history: list, length = number of trained epochs, each element has shape (n,)
        age_history_np = np.stack(age_history, axis=0)
        # Average Age of each sample across all epochs: shape (n,)
        avg_age_per_sample = np.nanmean(age_history_np, axis=0)

        # Corresponding training-set labels (note that IndexedDataset wraps the dataset)
        labels = np.array(train_eval_set.base.targets, dtype=np.int64)

        avg_age_per_class = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            mask = (labels == c) & np.isfinite(avg_age_per_sample)
            if mask.any():
                avg_age_per_class[c] = avg_age_per_sample[mask].mean()
            else:
                avg_age_per_class[c] = np.nan  # should not happen in theory, but kept for safety
    else:
        # online: directly average the class-level age over each epoch
        online_class_age_history_np = np.stack(online_class_age_history, axis=0)  # (E, C)
        avg_age_per_class = np.mean(online_class_age_history_np, axis=0)

    # ======= (Added) Test set: per-sample average Age + per-class average Age =======
    age_history_test_np = np.stack(age_history_test, axis=0)  # (E, n_test)
    avg_age_per_sample_test = age_history_test_np.mean(axis=0)

    test_labels = np.array(test_set.targets, dtype=np.int64)  # Note that test_set.targets uses the original label order
    avg_age_per_class_test = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = (test_labels == c)
        if mask.any():
            avg_age_per_class_test[c] = avg_age_per_sample_test[mask].mean()
        else:
            avg_age_per_class_test[c] = np.nan

    # ======= After training: compute per-class accuracy of the best and last models on the test set =======
    # If best_path is still None in an extreme case (should not happen in theory), fall back to the last model
    if best_path is None:
        best_path = last_path
        logger.warning("best_path is None, fallback to last model as best model.")

    # 1) best model (reloaded from best_path)
    # best_model = ResNet34_CIFAR(num_classes=100).to(device)
    # best_model.load_state_dict(torch.load(best_path, map_location=device))
    # best_model = resnet32(num_classes=100, use_norm=False).to(device)
    # best_model = ResNetCifarTorch(
    #     num_layers=32,
    #     num_classes=num_classes,
    #     bn_decay=0.9,
    #     bn_eps=1e-5,
    #     loss_type=args.loss_type
    # ).to(device)
    # best_model.load_state_dict(torch.load(best_path, map_location=device))
    best_model = build_model(args, num_classes).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    per_class_acc_best = per_class_accuracy(best_model, test_loader, device, num_classes=num_classes)

    # ======= (Added) At the end of training: log the best model's per-class accuracy on the test set =======
    logger.info("===== BEST model per-class accuracy on TEST set =====")
    for c in range(num_classes):
        logger.info(f"class {c:03d}: acc_best={per_class_acc_best[c]:.6f}, test_count={int(test_class_count[c])}")
    logger.info("===== END per-class accuracy =====")

    # 2) last model (the current model)
    per_class_acc_last = per_class_accuracy(model, test_loader, device, num_classes=num_classes)

    per_class_avg_margin_last, per_class_avg_loss_last = \
        per_class_margin_and_loss(
            model,
            test_loader,
            device,
            num_classes=num_classes
        )

    # ======= Write the new per-class statistics CSV =======
    os.makedirs("per_class_stats", exist_ok=True)
    class_csv_path = os.path.join("per_class_stats", f"{run_name}_per_class_age_acc_counts.csv")

    with open(class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # header:class_idx, avg_age, acc_best, acc_last, train_count, test_count
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

    # ===== (Added) Log footer: print all parameters and values for this run =====
    logger.info(f"RUN_ARGS: {args_kv_line}")

    # ===== (Added) Log footer: output final test_acc / best_acc + timestamp (month to second) in your specified format =====
    # Note: test_acc here is the test_acc from the last epoch; best_acc is the best value over the entire run
    logger.info(f"{test_acc:.4f} ({best_acc:.4f}), {ts_md_hms}")


if __name__ == '__main__':
    main()
