import random
import time
from collections import Counter
from glob import glob
import os
from os.path import join

import numpy as np
import scaper
import argparse

import yaml

from dataset.data_utils import get_class_assignments, get_source_path_splits
from dataset.soundscape_generation import (
    create_soundscape,
    generate_without_audio,
    SEED,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fgpath", type=str, required=True, help="path to foreground source files"
    )
    parser.add_argument(
        "--bgpath",
        type=str,
        required=False,
        help="path to background files, not required for OST",
    )
    parser.add_argument(
        "--outpath", type=str, required=True, help="base path to save output jams files"
    )
    parser.add_argument(
        "--nvariants",
        type=int,
        required=True,
        help="generate up to and including this dataset variant [1 - 5]",
    )
    parser.add_argument(
        "--openness", type=str, required=True, help="openness: high or low"
    )

    args = parser.parse_args()

    assert (args.nvariants < 6) & (args.nvariants > 0)
    assert os.path.isdir(args.fgpath)
    assert os.path.isdir(args.bgpath)
    assert args.openness in ["high", "low"]

    return args


def generate_split(
    sc,
    n,
    split,
    n_split_soundscapes,
    split_source_paths,
    split_source_counts,
    split_class_idx,
    outpath,
    config,
):
    """
    Generate a specific dataset variant split and save JAMS files

    Params
    -------
    sc: Scaper Soundscape object
    n: starting number for file names

    Returns
    -------
    sc object
    """
    for class_id in split_class_idx:
        for i in range(int(config["min_examples_per_class"])):
            sc = create_soundscape(
                sc,
                split_source_paths,
                split_source_counts,
                class_id,
                snr_min=float(config["clean_snr"]),
                snr_max=float(config["clean_snr"]),
                add_bg=bool(config["add_bg"]),
            )

            jamsfile = join(outpath, split, f"{n}.jams")
            generate_without_audio(sc, jamsfile)
            n += 1

    n_class_wise_soundscapes = len(split_class_idx) * int(
        config["min_examples_per_class"]
    )
    for i in range(n_split_soundscapes - n_class_wise_soundscapes):
        sc = create_soundscape(
            sc,
            split_source_paths,
            split_source_counts,
            class_id=None,
            snr_min=float(config["clean_snr"]),
            snr_max=float(config["clean_snr"]),
            add_bg=bool(config["add_bg"]),
        )

        jamsfile = join(outpath, split, f"{n}.jams")
        generate_without_audio(sc, jamsfile)
        n += 1
    return sc


def main():
    args = parse_args()

    with open(join("dataset", "oss.yml"), "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config file")
    print(config)

    # get shuffled class ids
    vocab_idx = [i for i in range(89)]
    random.Random(SEED).shuffle(vocab_idx)

    # get paths
    source_paths = glob(join(args.fgpath, "*/*.wav"))

    # get source class counts
    source_class_counts = Counter([path.split("/")[-2] for path in source_paths])
    start_time = time.time()

    sc = scaper.Scaper(
        duration=int(config["duration"]),
        fg_path=args.fgpath,
        bg_path=args.bgpath,
        random_state=SEED,
    )
    sc.sr = int(config["sr"])
    sc.n_channels = int(config["n_channels"])
    sc.ref_db = float(config["ref_db"])

    np.random.seed(SEED)

    # generate min_examples_per_class for each fold
    for variant_id in range(1, args.nvariants + 1):
        print(f"Generating examples for variant {variant_id}")
        n = 0
        dataset_variant_outpath = join(
            args.outpath, args.openness, f"variant{variant_id}"
        )
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(dataset_variant_outpath, split), exist_ok=True)

        # get kk, ku and uu classes
        kk_idx, ku_idx, uu_idx = get_class_assignments(variant_id, vocab_idx)
        if args.openness == "high":
            known_classes = kk_idx
            unknown_classes = ku_idx + uu_idx
        else:
            known_classes = kk_idx + ku_idx
            unknown_classes = uu_idx

        # get paths
        train_paths, val_paths, test_paths = get_source_path_splits(
            source_paths,
            known_classes,
            unknown_classes,
            train_frac=float(config["train_paths_frac"]),
            val_frac=float(config["val_paths_frac"]),
        )

        # train split
        train_source_counts = Counter(
            {str(k): source_class_counts.get(str(k), 0) for k in kk_idx + ku_idx}
        )

        sc = generate_split(
            sc,
            0,
            "train",
            int(config["n_train_soundscapes"]),
            train_paths,
            train_source_counts,
            known_classes,
            dataset_variant_outpath,
            config,
        )
        print("Generated training set examples")

        # val split
        val_source_counts = train_source_counts  # same classes seen in train and val
        sc = generate_split(
            sc,
            int(config["n_train_soundscapes"]),
            "val",
            int(config["n_val_soundscapes"]),
            val_paths,
            val_source_counts,
            known_classes,
            dataset_variant_outpath,
            config,
        )

        print("Generated validation set examples")

        # test split
        test_source_counts = source_class_counts
        sc = generate_split(
            sc,
            int(config["n_train_soundscapes"]) + int(config["n_val_soundscapes"]),
            "test",
            int(config["n_test_soundscapes"]),
            test_paths,
            test_source_counts,
            known_classes + unknown_classes,
            dataset_variant_outpath,
            config,
        )

        print("Generated testing set examples")

        print(
            f"Synthesized training, validation and test splits for variant {variant_id}--"
        )
        print(f"Training: {config['n_train_soundscapes']} ")
        print(f"Validation: {config['n_val_soundscapes']} ")
        print(f"Test: {config['n_test_soundscapes']} ")
        print("--------------------------------------------------------")

    print(
        f"Synthesized dataset in {round((time.time() - start_time) / 60.0, 2)} minutes"
    )


if __name__ == "__main__":
    main()
