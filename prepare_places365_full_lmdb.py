#!/usr/bin/env python3

import argparse
import io
import json
import multiprocessing as mp
import os
import struct
import tarfile

import lmdb
import numpy as np
from PIL import Image

TRAIN_SIZE = 1803460
VAL_SIZE = 36500
NUM_CLASSES = 365


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def load_filelist(path, expected_size):
    path_to_idx = {}
    targets = np.empty(expected_size, dtype=np.int16)

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            image_path, label_text = line.rsplit(maxsplit=1)
            label = int(label_text)

            # Normalize both train and val official paths.
            key = image_path.lstrip("/")

            if key in path_to_idx:
                raise RuntimeError(f"Duplicate path in filelist: {key}")

            if not (0 <= label < NUM_CLASSES):
                raise RuntimeError(f"Invalid label {label}: {key}")

            if idx >= expected_size:
                raise RuntimeError(
                    f"Filelist has more than {expected_size} entries"
                )

            path_to_idx[key] = idx
            targets[idx] = label

    if len(path_to_idx) != expected_size:
        raise RuntimeError(
            f"Filelist size mismatch: "
            f"{len(path_to_idx)} != {expected_size}"
        )

    return path_to_idx, targets


def encode_image(job):
    idx, label, raw, image_size, jpeg_quality = job

    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )

        buf = io.BytesIO()
        image.save(
            buf,
            format="JPEG",
            quality=jpeg_quality,
            optimize=False,
        )

    encoded = struct.pack(">H", int(label)) + buf.getvalue()
    return idx, encoded


def normalize_member_name(split, member_name):
    name = member_name.lstrip("./")

    if split == "train":
        prefix = "data_large/"
        if not name.startswith(prefix):
            return None
        return name[len(prefix):]

    if split == "val":
        prefix = "val_large/"
        if not name.startswith(prefix):
            return None

        name = name[len(prefix):]
        if not name:
            return None

        return name

    raise ValueError(split)


def convert_split(
        split,
        tar_path,
        list_path,
        output_root,
        expected_size,
        image_size,
        jpeg_quality,
        workers,
        map_size,
):
    print(f"===== {split.upper()} =====", flush=True)
    print(f"tar      = {tar_path}", flush=True)
    print(f"filelist = {list_path}", flush=True)

    path_to_idx, targets = load_filelist(
        list_path,
        expected_size,
    )

    counts = np.bincount(
        targets.astype(np.int64),
        minlength=NUM_CLASSES,
    )

    print(
        f"class count min/max = {counts.min()}/{counts.max()}",
        flush=True,
    )

    lmdb_path = os.path.join(output_root, f"{split}.lmdb")
    targets_path = os.path.join(
        output_root,
        f"{split}_targets.npy",
    )

    if os.path.exists(lmdb_path):
        raise RuntimeError(f"Already exists: {lmdb_path}")

    if os.path.exists(targets_path):
        raise RuntimeError(f"Already exists: {targets_path}")

    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_readers=512,
    )

    seen = np.zeros(expected_size, dtype=np.bool_)

    def jobs():
        with tarfile.open(tar_path, mode="r:") as tf:
            for member in tf:
                if not member.isfile():
                    continue

                key = normalize_member_name(
                    split,
                    member.name,
                )

                if key is None:
                    continue

                if key not in path_to_idx:
                    if key.lower().endswith(".jpg"):
                        raise RuntimeError(
                            f"Image in tar missing from official filelist: "
                            f"{member.name}"
                        )
                    continue

                idx = path_to_idx[key]
                label = int(targets[idx])

                f = tf.extractfile(member)
                if f is None:
                    raise RuntimeError(
                        f"Cannot read tar member: {member.name}"
                    )

                raw = f.read()

                yield (
                    idx,
                    label,
                    raw,
                    image_size,
                    jpeg_quality,
                )

    txn = env.begin(write=True)
    pending = 0
    done = 0

    with mp.Pool(processes=workers) as pool:
        for idx, encoded in pool.imap(
                encode_image,
                jobs(),
                chunksize=16,
        ):
            if seen[idx]:
                raise RuntimeError(
                    f"Duplicate image encountered for idx={idx}"
                )

            key = struct.pack(">Q", int(idx))
            txn.put(key, encoded)

            seen[idx] = True
            pending += 1
            done += 1

            if pending >= 5000:
                txn.commit()
                txn = env.begin(write=True)
                pending = 0

            if done % 10000 == 0:
                print(
                    f"[{split}] {done}/{expected_size}",
                    flush=True,
                )

    txn.commit()
    env.sync()
    env.close()

    if done != expected_size:
        missing = np.flatnonzero(~seen)
        raise RuntimeError(
            f"{split}: converted {done}/{expected_size}; "
            f"missing count={len(missing)}, "
            f"first missing={missing[:20].tolist()}"
        )

    if not np.all(seen):
        raise RuntimeError(f"{split}: seen mask incomplete")

    np.save(targets_path, targets)

    print(
        f"[{split}] conversion complete: {done}",
        flush=True,
    )
    print(f"[{split}] LMDB    = {lmdb_path}", flush=True)
    print(f"[{split}] targets = {targets_path}", flush=True)


def verify_split(lmdb_path, targets_path, expected_size):
    targets = np.load(targets_path)

    if len(targets) != expected_size:
        raise RuntimeError(
            f"targets length {len(targets)} != {expected_size}"
        )

    env = lmdb.open(
        lmdb_path,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=512,
    )

    entries = env.stat()["entries"]

    if entries != expected_size:
        raise RuntimeError(
            f"LMDB entries {entries} != {expected_size}"
        )

    check_indices = [
        0,
        expected_size // 4,
        expected_size // 2,
        3 * expected_size // 4,
        expected_size - 1,
    ]

    with env.begin(write=False) as txn:
        for idx in check_indices:
            key = struct.pack(">Q", idx)
            encoded = txn.get(key)

            if encoded is None:
                raise RuntimeError(f"Missing LMDB key {idx}")

            label = struct.unpack(">H", encoded[:2])[0]

            if label != int(targets[idx]):
                raise RuntimeError(
                    f"Label mismatch idx={idx}: "
                    f"{label} != {targets[idx]}"
                )

            with Image.open(io.BytesIO(encoded[2:])) as image:
                image.load()

                if image.size != (224, 224):
                    raise RuntimeError(
                        f"Image size mismatch idx={idx}: "
                        f"{image.size}"
                    )

    env.close()

    print(
        f"VERIFY OK: {lmdb_path}, entries={entries}",
        flush=True,
    )


def main():
    args = parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    train_tar = os.path.join(
        args.archive_root,
        "train_large_places365standard.tar",
    )
    val_tar = os.path.join(
        args.archive_root,
        "val_large.tar",
    )

    train_list = os.path.join(
        args.archive_root,
        "places365_train_standard.txt",
    )
    val_list = os.path.join(
        args.archive_root,
        "places365_val.txt",
    )

    for path in [
        train_tar,
        val_tar,
        train_list,
        val_list,
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    convert_split(
        split="train",
        tar_path=train_tar,
        list_path=train_list,
        output_root=args.output_root,
        expected_size=TRAIN_SIZE,
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        map_size=64 * 1024 ** 3,
    )

    convert_split(
        split="val",
        tar_path=val_tar,
        list_path=val_list,
        output_root=args.output_root,
        expected_size=VAL_SIZE,
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        map_size=4 * 1024 ** 3,
    )

    verify_split(
        os.path.join(args.output_root, "train.lmdb"),
        os.path.join(args.output_root, "train_targets.npy"),
        TRAIN_SIZE,
    )

    verify_split(
        os.path.join(args.output_root, "val.lmdb"),
        os.path.join(args.output_root, "val_targets.npy"),
        VAL_SIZE,
    )

    train_targets = np.load(
        os.path.join(args.output_root, "train_targets.npy")
    )
    val_targets = np.load(
        os.path.join(args.output_root, "val_targets.npy")
    )

    train_counts = np.bincount(
        train_targets.astype(np.int64),
        minlength=NUM_CLASSES,
    )
    val_counts = np.bincount(
        val_targets.astype(np.int64),
        minlength=NUM_CLASSES,
    )

    metadata = {
        "dataset": "Places365-Standard",
        "source": "official high-resolution Places365-Standard archives",
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "num_classes": NUM_CLASSES,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
        "train_class_min": int(train_counts.min()),
        "train_class_max": int(train_counts.max()),
        "val_class_min": int(val_counts.min()),
        "val_class_max": int(val_counts.max()),
    }

    with open(
            os.path.join(args.output_root, "metadata.json"),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("===== ALL DONE =====", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
