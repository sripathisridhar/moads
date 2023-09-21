import argparse
import time
import jams
import os
from os import path
from pathlib import Path
from glob import glob
import soundfile as sf
import scaper
import yaml
import librosa
import numpy as np


def get_failed_jams(
    openness,
    fold,
    split,
    jams_dir="/research/mc232/sound_datasets/oss-clean/jams/",
    log_dir="/home/s/ss645/mlos/out/ground_truth/",
):
    jams_dir = path.join(jams_dir, openness, fold, split)
    try:
        with open(path.join(log_dir, f"{openness}.{fold}.{split}.txt"), "r") as f:
            data = f.readlines()
            data = [
                path.join(jams_dir, path.basename(line).split("_")[-3] + ".jams")
                for line in data
            ]
    except Exception as e:
        print(e)
        return []
    print("Number of lines in log file=", len(data))
    data = list(set(data))
    print("Number of jams files for resynthesis=", len(data))
    return data


def ground_truth_estimates(
    file_list,
    split_dir=None,
    target_sr=16000,
    label_in_file_name=True,
    out_dir_id="ost-clean-gt",
    jams_dir_id="oss-clean",
):
    # file_list = glob(path.join(split_dir, "*.jams"))
    for file in file_list:
        # grab foreground event annotations
        jams_dump = jams.load(file).search(namespace="scaper")[0]
        orig_sr, duration = (
            jams_dump["sandbox"].scaper["sr"],
            jams_dump["sandbox"].scaper["duration"],
        )
        anns = jams_dump.data
        anns = [event for event in anns if event.value["label"] != "brownnoise"]
        # get list of event arrays
        _, _, _, event_audio_list = scaper.generate_from_jams(file)
        event_audio_list = librosa.resample(
            np.array(event_audio_list).squeeze(axis=-1),
            orig_sr=orig_sr,
            target_sr=target_sr,
        )
        assert event_audio_list.shape[-1] == (target_sr * duration)

        # if "clean" not in out_dir_id:
        #     event_audio_list = event_audio_list[1:]  # ignore background noise
        openness, fold, split = split_dir.split("/")[-3:]

        labels, start_times, end_times = [], [], []
        for event in anns:
            labels.append(event.value["label"])
            start_times.append(event.time)
            end_times.append(event.time + event.duration)

        for i, (ann, event_wav) in enumerate(zip(anns, event_audio_list)):
            clip_basepath = (
                path.splitext(file)[0]
                .replace("jams", "audio")
                .replace(jams_dir_id, out_dir_id)
                + f"_{i+1}"
            )
            start_time = ann.time
            end_time = start_time + ann.duration
            mid_point = (start_time + end_time) / 2

            # crop middle 1s of the event
            start_time = mid_point - 0.5
            if start_time < 0:
                start_time = 0
            elif (start_time + 1) >= duration:
                extra = (start_time + 1) - duration
                start_time = start_time - extra
            end_time = start_time + 1

            event_wav = event_wav[
                int(start_time * target_sr) : int(end_time * target_sr)
            ]
            if label_in_file_name:
                event_out_path = clip_basepath + f"_{labels[i]}.wav"
            else:
                event_out_path = clip_basepath + f"_1.wav"

            os.makedirs(path.dirname(event_out_path), exist_ok=True)
            try:
                sf.write(event_out_path, event_wav, samplerate=target_sr)
            except Exception as e:
                # print(e)
                with open(
                    f"/home/s/ss645/mlos/out/ost-clean-gt/{openness}.{fold}.{split}.txt",
                    "a",
                ) as f:
                    f.write(e, event_out_path + "\n")

            other_idx = [k for k in range(len(anns))]
            other_idx.pop(i)
            overlap_idx = 1
            for j in other_idx:
                if (
                    (start_time < start_times[j] < end_time)
                    or (start_time < end_times[j] < end_time)
                    or ((start_times[j] < start_time) and (end_times[j] > end_time))
                ):
                    overlap_idx += 1
                    event_wav = event_audio_list[j][
                        int(start_time * target_sr) : int(end_time * target_sr)
                    ]
                    if label_in_file_name:
                        event_out_path = clip_basepath + f"_{labels[j]}.wav"
                    else:
                        event_out_path = clip_basepath + f"_{overlap_idx}.wav"
                    try:
                        sf.write(event_out_path, event_wav, samplerate=target_sr)
                    except Exception as e:
                        with open(
                            f"/home/s/ss645/mlos/out/ost-clean-gt/{openness}.{fold}.{split}.txt",
                            "a",
                        ) as f:
                            f.write(e, event_out_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--openness", help="\{high, low\}")
    parser.add_argument("-f", "--variant", help="variant\{1,2,..,5\}")
    parser.add_argument("-s", "--split", help="\{train, val, test\}")
    args = parser.parse_args()

    with open("tag.yml", "r") as f:
        config = yaml.safe_load(f)
    print(config)

    # file_list = get_failed_jams(args.openness, args.fold, args.split)
    # if len(file_list) == 0:
    #     print("This split has no pending files, exiting now")
    # else:
    #     assert all([path.isfile(file) for file in file_list])

    start_time = time.time()
    split_dir = path.join(config["jams_dir"], args.openness, args.fold, args.split)
    print(split_dir)
    paths = [str(path) for path in Path(split_dir).rglob("*.jams")]
    print(
        f"Generating from {len(paths)} jams files for {args.openness}, {args.fold}, {args.split}"
    )
    ground_truth_estimates(
        paths,
        split_dir,
        config["sr"],
        config["label_in_file_name"],
        config["out_dir_id"],
        config["jams_dir_id"],
    )
    split_time = time.time() - start_time
    print(f"Generated the split in {split_time} s")
    print("-------------------------------------------------")
