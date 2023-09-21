import os
from pathlib import Path
import time
import numpy as np
import jams
import scaper
import librosa
import soundfile as sf
import pandas as pd
import argparse


def create_tag(
    split_dir,
    generate_audio,
    target_sr,
    out_dir_id,
    jams_dir_id,
    save_isolated_events=False,
    gt_dir_id=None,
):
    """
    Create the tag dataset based on the given directory of jams files
    Params:
    ------
    split_dir : path to directory with jams files
    generate_audio : If True, generate Tag audio files as well
    """

    # if "train" in split_dir:
    #     paths = glob.glob(os.path.join(split_dir, "*.jams"))
    # else:
    #     paths = (
    #         glob.glob(os.path.join(split_dir, "uu/*.jams"))
    #         + glob.glob(os.path.join(split_dir, "kk/seen/*.jams"))
    #         + glob.glob(os.path.join(split_dir, "kk/unseen/*.jams"))
    #     )
    paths = [str(path) for path in Path(split_dir).rglob("*.jams")]

    columns = ["file_name", "source_file", "start_time", "label"]
    df = pd.DataFrame(columns=columns)
    index = 0
    pkl_dir = os.path.join(
        split_dir.replace(jams_dir_id, out_dir_id).split("jams")[0], "ann"
    )
    os.makedirs(pkl_dir, exist_ok=True)
    openness, variant_id, split = split_dir.split("/")[-3:]

    dirs = set(
        [
            str(path.parent).replace(jams_dir_id, out_dir_id)
            for path in Path(split_dir).rglob("*.jams")
        ]
    )
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    for jamsPath in paths:
        fName = os.path.splitext(jamsPath)[0].replace("jams", "audio") + ".wav"

        jamsFile = jams.load(jamsPath)

        # if soundscape wav file doesn't exist, generate audio array
        if not os.path.isfile(fName):
            try:
                audioArray, _, _, _ = scaper.generate_from_jams(jamsPath, None)
            except:
                with open(
                    f"/home/s/ss645/mlos/logs/{openness}.{variant_id}.{split}.txt", "a"
                ) as f:
                    f.write("Scaper:" + jamsPath + "\n")
                continue
        else:
            try:
                audioArray, _ = librosa.load(fName, sr=target_sr)
            except:
                with open(
                    f"/home/s/ss645/mlos/logs/{openness}.{variant_id}.{split}.txt", "a"
                ) as f:
                    f.write("Librosa:" + jamsPath + "\n")
                continue

        annotations = jamsFile.annotations.search(namespace="scaper")[0].data
        annotations = [
            event for event in annotations if event.value["label"] != "brownnoise"
        ]
        eventCount = len(annotations)

        labels = []
        start_times = []
        end_times = []
        for event in annotations:
            labels.append(event.value["label"])
            start_times.append(event.time)
            end_times.append(event.time + event.duration)

        for i in range(eventCount):
            fileLabel = []
            fileLabel.append(labels[i])
            startTime = start_times[i]
            endTime = end_times[i]

            midPoint = (startTime + endTime) / 2
            startTime = midPoint - 0.5  # window start time

            if startTime < 0:
                startTime = 0
            elif (
                startTime + 1
            ) >= 10:  # probably not needed as latest event start time is 9s
                extra = (startTime + 1) - 10
                startTime = startTime - extra
            endTime = startTime + 1  # window end time

            other_idx = [k for k in range(eventCount)]
            other_idx.pop(i)
            for j in other_idx:
                if (
                    (startTime < start_times[j] < endTime)
                    or (startTime < end_times[j] < endTime)
                    or ((start_times[j] < startTime) and (end_times[j] > endTime))
                ):
                    fileLabel.append(labels[j])

            sampleStart = int(startTime * target_sr)

            eventArray = audioArray[sampleStart : sampleStart + target_sr]

            trimfName = fName.replace(".wav", "_" + str(i + 1) + ".wav").replace(
                jams_dir_id, out_dir_id
            )

            if generate_audio:
                try:
                    sf.write(trimfName, eventArray, target_sr)
                except:
                    with open(
                        f"/home/s/ss645/mlos/out/{openness}.{variant_id}.{split}.txt",
                        "a",
                    ) as f:
                        f.write(trimfName + "\n")

            df2 = pd.DataFrame.from_dict(
                {index: [trimfName, jamsPath.split("/")[-1], startTime, fileLabel]},
                orient="index",
                columns=columns,
            )
            df = pd.concat([df, df2], ignore_index=True, axis=0)
            index = index + 1

    df.to_pickle(os.path.join(pkl_dir, f"{openness}_{variant_id}_{split}.pkl"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--openness", type=str, required=True, help="\{high, low\}"
    )
    parser.add_argument(
        "-v", "--variant", type=str, required=True, help="variant\{1,2,..,5\}"
    )
    parser.add_argument(
        "-s", "--split", type=str, required=True, help="\{train, val, test\}"
    )
    parser.add_argument(
        "-p",
        "--osspath",
        type=str,
        required=True,
        help="path to base directory with openness, dataset variants, and splits",
    )
    parser.add_argument(
        "--jamid",
        type=str,
        required=False,
        help="Name of jams dataset, e.g. oss",
        default="oss",
    )
    parser.add_argument(
        "--outid",
        type=str,
        required=False,
        help="Name of jams dataset, e.g. oss",
        default="ost",
    )
    parser.add_argument(
        "--sr",
        type=int,
        required=False,
        help="sample rate of output wav files",
        default=16_000,
    )
    parser.add_argument(
        "--genaudio",
        type=bool,
        required=False,
        help="whether to save wav files. If false, just save annotation file",
        default=True,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    print(
        f"Generating from openness {args.openness}, {args.variant}, {args.split} split"
    )
    print(args)

    start_time = time.time()
    split_dir = os.path.join(args.osspath, args.openness, args.variant, args.split)
    print(split_dir)
    create_tag(
        split_dir,
        args.genaudio,
        args.sr,
        args.outid,
        args.jamid,
    )
    split_time = time.time() - start_time
    print(f"Generated the split in {round(split_time / 60.0, 2)} minutes")
    print("-------------------------------------------------")
