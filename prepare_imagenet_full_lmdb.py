#!/usr/bin/env python3

import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import struct

import lmdb
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def extract_image_bytes(image_value):
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        if image_bytes is not None:
            return image_bytes

        path = image_value.get("path")
        if path is not None and os.path.isfile(path):
            with open(path, "rb") as f:
                return f.read()

    if isinstance(image_value, (bytes, bytearray, memoryview)):
        return bytes(image_value)

    raise RuntimeError(f"Unsupported image representation: {type(image_value)}")


def process_one(item):
    idx, image_value, label = item

    image_bytes = extract_image_bytes(image_value)

    # Verify that PIL can decode the source image, but keep the original
    # compressed bytes unchanged. Resize/crop is done during training.
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()

    encoded = struct.pack(">H", int(label)) + image_bytes

    return idx, int(label), encoded


def convert_split(split, files, output_root, workers):
    print(f"[{split}] shards = {len(files)}", flush=True)

    total_rows = 0
    for path in files:
        pf = pq.ParquetFile(path)
        total_rows += pf.metadata.num_rows

    print(f"[{split}] total images = {total_rows}", flush=True)

    lmdb_path = os.path.join(output_root, f"{split}.lmdb")
    targets_path = os.path.join(output_root, f"{split}_targets.npy")

    os.makedirs(lmdb_path, exist_ok=True)

    map_size = 256 * 1024 ** 3 if split == "train" else 16 * 1024 ** 3

    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False
    )

    targets = np.empty(total_rows, dtype=np.int16)

    global_idx = 0
    txn = env.begin(write=True)
    pending = 0

    with mp.Pool(processes=workers) as pool:
        for shard_idx, path in enumerate(files):
            print(
                f"[{split}] shard {shard_idx + 1}/{len(files)}: "
                f"{os.path.basename(path)}",
                flush=True
            )

            parquet_file = pq.ParquetFile(path)

            for batch in parquet_file.iter_batches(
                    batch_size=512,
                    columns=["image", "label"]
            ):
                image_values = batch.column(
                    batch.schema.get_field_index("image")
                ).to_pylist()

                labels = batch.column(
                    batch.schema.get_field_index("label")
                ).to_pylist()

                jobs = [
                    (
                        global_idx + i,
                        image_values[i],
                        int(labels[i]),
                    )
                    for i in range(len(labels))
                ]

                for idx, label, encoded in pool.imap(
                        process_one,
                        jobs,
                        chunksize=16
                ):
                    key = struct.pack(">Q", idx)
                    txn.put(key, encoded)
                    targets[idx] = label
                    pending += 1

                    if pending >= 5000:
                        txn.commit()
                        txn = env.begin(write=True)
                        pending = 0

                global_idx += len(labels)

    txn.commit()
    env.sync()
    env.close()

    if global_idx != total_rows:
        raise RuntimeError(
            f"{split}: processed {global_idx}, expected {total_rows}"
        )

    np.save(targets_path, targets)

    print(f"[{split}] finished: {global_idx} images", flush=True)
    print(f"[{split}] LMDB: {lmdb_path}", flush=True)
    print(f"[{split}] targets: {targets_path}", flush=True)


def main():
    args = parse_args()

    data_root = os.path.join(args.input_root, "data")

    train_files = sorted(
        glob.glob(os.path.join(data_root, "train-*.parquet"))
    )

    val_files = sorted(
        glob.glob(os.path.join(data_root, "validation-*.parquet"))
    )

    if len(train_files) != 294:
        raise RuntimeError(
            f"Expected 294 ImageNet train shards, found {len(train_files)}"
        )

    if len(val_files) != 14:
        raise RuntimeError(
            f"Expected 14 ImageNet validation shards, found {len(val_files)}"
        )

    os.makedirs(args.output_root, exist_ok=True)

    convert_split(
        "train",
        train_files,
        args.output_root,
        args.workers
    )

    convert_split(
        "val",
        val_files,
        args.output_root,
        args.workers
    )

    metadata = {
        "dataset": "ImageNet-1K",
        "train_size": 1281167,
        "val_size": 50000,
        "num_classes": 1000,
        "image_storage": "original compressed bytes",
        "training_input_size": 224,
        "source": "ILSVRC/imagenet-1k Hugging Face parquet"
    }

    with open(
            os.path.join(args.output_root, "metadata.json"),
            "w",
            encoding="utf-8"
    ) as f:
        f.write(json.dumps(metadata, indent=2))

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
