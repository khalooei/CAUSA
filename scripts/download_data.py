"""Downloads and extracts the BEAF object-existence audio archive.

The benchmark rows are streamed by the Hugging Face `datasets` library
(see `causa.data.DATASET_ID`); only the audio, hosted as a single tar
archive, needs this separate download.

Usage:
    python scripts/download_data.py
"""
import argparse
import os
import tarfile
import urllib.request

TAR_URL = "https://huggingface.co/datasets/kuanhuggingface/BEAF-Audio/resolve/main/BEAF_Audio.tar"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "BEAF_Audio.tar")
    audio_dir = os.path.join(data_dir, "audio")

    if not os.path.exists(tar_path):
        print(f"downloading {TAR_URL} -> {tar_path}")
        urllib.request.urlretrieve(TAR_URL, tar_path)
    else:
        print(f"found existing {tar_path}, skipping download")

    if not os.path.isdir(audio_dir) or not os.listdir(audio_dir):
        print(f"extracting {tar_path} -> {audio_dir}")
        os.makedirs(audio_dir, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(audio_dir, filter="data")
    else:
        print(f"found existing extracted audio at {audio_dir}, skipping extraction")

    # the tar's own top-level entry is a "beaf_audio/" directory holding
    # the .wav files that dataset rows' `audio` field refers to directly
    audio_root = os.path.join(audio_dir, "beaf_audio")
    print("done. audio_root =", audio_root)


if __name__ == "__main__":
    main()
