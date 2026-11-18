"""Acquire public THINGS-EEG2 pilot inputs from pinned sources."""
import argparse
from pathlib import Path
import shutil
from urllib.request import urlopen

from .things import EEG, EVENT_HASHES, IMAGES, LICENSE, MANIFEST, METADATA, checksum

OSF = "https://osf.io/download/"
BIDS = "https://raw.githubusercontent.com/nemarDatasets/nm000232/cf0540ffd250d8681ef31a9a9e345f5053e4c90b/"



def fetch(url, path, expected):
    if path.exists():
        checksum(path, expected)
        return
    print(f"Downloading {path}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, 2**20)
    checksum(temporary, expected)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research-data/things-eeg2"))
    parser.add_argument("--subjects", nargs="+", choices=("01", "02"), default=["01"])
    args = parser.parse_args()
    root = args.root
    fetch(OSF + "3765v/", root / "IMAGE-LICENSE.txt",
          LICENSE)
    fetch(OSF + "znu7b/", root / "test_images.zip", IMAGES)
    fetch(OSF + "qkgtf/", root / "image_metadata.npy", METADATA)
    fetch(BIDS + "stimuli/stimuli.tsv", root / "stimuli.tsv",
          MANIFEST)
    files = {"01": "q57yf", "02": "684f5bc711de432f5ab78752"}
    for subject in args.subjects:
        folder = root if subject == "01" else root / f"sub-{subject}"
        fetch(f"{OSF}{files[subject]}/", folder / "test.npy", EEG[subject])
        for session, expected in enumerate(EVENT_HASHES[subject], 1):
            label = f"sub-{subject}_ses-{session:02d}_task-test_events.tsv"
            fetch(f"{BIDS}sub-{subject}/ses-{session:02d}/eeg/{label}",
                  folder / f"events-{session:02d}.tsv", expected)


if __name__ == "__main__":
    main()
