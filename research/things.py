"""Load only checksum-pinned public THINGS-EEG2 pilot derivatives."""
from collections import Counter
import csv
import hashlib
from pathlib import Path
import zipfile

import cv2
import numpy as np

from .cove import extract

EEG = {
    "01": "b8c1af38067b9a861272d7259b0c814409c8258a02207bf923870d7e3f8f362a",
    "02": "d50716185511109320580676e6160f54fc4b62db9ac971ecf7e142d3b2edbebb",
}
METADATA = "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c"
IMAGES = "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a"
LICENSE = "ea94c104d2319aa65d82f4b179531141d3ec1483547826c3a72cfd45015673dc"
MANIFEST = "9669cd5ec4cb544e82e74c5fbd80b88e60197c30c663768dfc684aa112a55715"
EVENT_HASHES = {
    "01": (
        "9b47be3426ffe33c30d722c7eedc58a970edf74a18afcfa18f39560e39da497d",
        "ecf442b780c36685e8a27c7de981b267547ba80dc32150a936c169546e67dd83",
        "2802112280518b59989a708a280e5c641473994729134482631b829dc50ddcb0",
        "aef7a79d71e621eaa9c30cc699e0ee17819cbda33fe2666f2fd603522cb34142",
    ),
    "02": (
        "5f0698d762fc5f8f822dc5b888db91bef612149d5ea460a891bd8ea56a1333c9",
        "e141ed196e35eeac4bf3f3812f8f73abbeb17ad7d418210f327e67735848180e",
        "49b91d25abb844ffd5d4ac6db7818017b2ac8ad2b5d6c8e8e42920ef9b9f15c0",
        "fb8bd1e40bc839daf09180a58b2ca8e4315afa1e54fc28da183720c9ea9c4373",
    ),
}
CHANNELS = ("O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8")


def checksum(path, expected=None):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected is not None and expected != actual:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return actual


def verify_release(root, subject):
    folder = root if subject == "01" else root / f"sub-{subject}"
    eeg_path, metadata_path, images_path = (folder / "test.npy",
                                            root / "image_metadata.npy", root / "test_images.zip")
    hashes = {p.name: checksum(p, expected) for p, expected in
              ((eeg_path, EEG[subject]), (metadata_path, METADATA), (images_path, IMAGES))}
    # These exact published NumPy dictionaries use pickle. Never unpickle user input.
    recording = np.load(eeg_path, allow_pickle=True).item()
    metadata = np.load(metadata_path, allow_pickle=True).item()
    if recording["args"].sub != int(subject):
        raise ValueError("EEG participant mismatch")
    fields = ("test_img_concepts", "test_img_files", "test_img_concepts_THINGS")
    for field in fields:
        if list(recording["img_metadata"][field]) != list(metadata[field]):
            raise ValueError(f"EEG condition order differs from original metadata: {field}")
    concepts, filenames, groups = (metadata[field] for field in fields)
    if not len(set(groups)) == len(set(concepts)) == len(set(filenames)) == 200:
        raise ValueError("Expected 200 unique images and concepts")
    manifest = root / "stimuli.tsv"
    hashes[manifest.name] = checksum(manifest, MANIFEST)
    with manifest.open() as stream:
        entries = {row["stimulus_id"]: row for row in csv.DictReader(stream, delimiter="\t")}
    for k, (concept, filename) in enumerate(zip(concepts, filenames, strict=True), 1):
        key = f"stim-test{k:03d}"
        row = entries[key]
        if (Path(row["filename"]).parent.name != concept or
                f'original filename "{filename}"' not in row["description"]):
            raise ValueError(f"Stimulus manifest mismatch: {key}")
    events = []
    for session in range(1, 5):
        path = folder / f"events-{session:02d}.tsv"
        hashes[path.name] = checksum(path, EVENT_HASHES[subject][session - 1])
        counts = Counter()
        with path.open() as stream:
            samples = []
            for row in csv.DictReader(stream, delimiter="\t"):
                samples.append(int(row["sample"]))
                if row["trial_type"] != "image":
                    continue
                k = int(row["tot_img_number"])
                if not (1 <= k <= len(concepts) and int(row["value"]) == k
                        and row["category_name"] == concepts[k - 1]):
                    raise ValueError(f"Event/image mismatch: {path}, sample {row['sample']}")
                counts[k] += 1
        if (set(counts) != set(range(1, 201)) or min(counts.values()) < 20
                or any(b <= a for a, b in zip(samples, samples[1:]))):
            raise ValueError(f"Unexpected event coverage or timing: {path}")
        events.append({"image_events": counts.total(), "other_events": len(samples) - counts.total()})
    return recording, concepts, filenames, np.array(groups), hashes, events


def load(root, subject):
    recording, concepts, filenames, groups, hashes, events = verify_release(Path(root), subject)
    eeg = recording["preprocessed_eeg_data"]
    times = recording["times"]
    hz = recording["args"].sfreq
    if eeg.shape != (200, 80, 63, 180) or hz != 200 or not np.allclose(
            times, np.arange(-20, 160) / hz):
        raise ValueError("Unexpected EEG dimensions or timebase")
    channels = [recording["ch_names"].index(ch) for ch in CHANNELS]
    epochs = eeg[:, :, channels, :]
    if not np.isfinite(epochs).all():
        raise ValueError("Nonfinite EEG")
    post = epochs[..., 20:140]
    # Use 20 ms bin means from 0–600 ms and per-trial, then averaged, log band powers.
    erp = post.mean(axis=1, dtype=np.float64).reshape(200, len(channels), 30, 4).mean(axis=-1)
    window = np.hanning(post.shape[-1])
    fft = np.fft.rfft(post * window, axis=-1)
    power = abs(fft)**2 / (hz * (window @ window))
    frequency = np.fft.rfftfreq(len(window), 1 / hz)
    bands = ((4, 8), (8, 13), (13, 30), (30, 45))
    spectrum = np.stack([np.log(np.maximum(1e-20, power[..., (frequency >= lo) &
                                                           (frequency < hi)].mean(axis=-1)))
                         .mean(axis=1) for lo, hi in bands], axis=-1)
    visuals = []
    with zipfile.ZipFile(Path(root) / "test_images.zip") as archive:
        for concept, filename in zip(concepts, filenames, strict=True):
            raw = archive.read(f"test_images/{concept}/{filename}")
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Undecodable image {filename}")
            visuals.append(extract(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None], fps=24))
    return (visuals, groups, erp, spectrum,
            {"hashes": hashes, "eeg_shape": list(eeg.shape), "events": events,
             "channels": list(CHANNELS), "band_hz": bands, "erp_bin_centers_ms": list(range(10, 600, 20))})
