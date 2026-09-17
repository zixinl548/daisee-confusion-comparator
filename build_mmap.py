#!/usr/bin/env python
"""
build_mmap.py — rebuild DAiSEE frame tensors as uint8 memory-maps.

Builds any of: train / val / trainval / test
Output dtype is uint8 in [0, 255] (NOT [0,1]), which is what
keras.applications.EfficientNetB0 expects as raw input.

    ##########################################################
    #  IMPORTANT — after switching to these uint8 mmaps you   #
    #  MUST REMOVE the Rescaling(255.0) layer from the model. #
    #  uint8 [0,255] -> .astype(float32) -> [0.0, 255.0],     #
    #  which already matches what EfficientNetB0 wants.       #
    #  Leaving Rescaling(255.0) in gives [0, 65025] — a new   #
    #  version of the exact bug you just fixed.               #
    ##########################################################

Usage
-----
    # 1. ALWAYS run this first. Builds nothing, just reports.
    python build_mmap.py --inspect

    # 2. Then build, one split at a time
    python build_mmap.py --split val      --workers 4
    python build_mmap.py --split test     --workers 4
    python build_mmap.py --split trainval --workers 4

Each build writes three files to OUT_DIR:
    <split>_X_u8.mmap          uint8, shape (N, 8, 224, 224, 3)
    <split>_labels4.npz        Boredom/Engagement/Confusion/Frustration, int32
    <split>_manifest.csv       row index -> ClipID -> source path
The manifest is what makes this reproducible — keep it.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("cv2 not found.  pip install opencv-python-headless --break-system-packages")

# ----------------------------------------------------------------------
# CONFIG — check these against your actual layout before running
# ----------------------------------------------------------------------
ROOT      = Path("/work/users/z/i/zixinl/Emotion")
VIDEO_DIR = ROOT / "DataSet"      # expects DataSet/Train, /Validation, /Test
LABEL_DIR = ROOT / "Labels"       # expects TrainLabels.csv etc.
OUT_DIR   = ROOT / "tensors_u8"

SEQ_LEN  = 8
IMG_SIZE = 224
VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv"}

# how each split maps onto DAiSEE's directories and label files
SPLITS = {
    "train":    [("Train",      "TrainLabels.csv")],
    "val":      [("Validation", "ValidationLabels.csv")],
    "test":     [("Test",       "TestLabels.csv")],
    "trainval": [("Train",      "TrainLabels.csv"),
                 ("Validation", "ValidationLabels.csv")],
}

LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def norm_id(s):
    """DAiSEE ClipIDs appear as '1100011002', '1100011002.avi', with spaces.
    Reduce everything to the bare digits so CSV and filesystem agree."""
    s = str(s).strip()
    for e in VIDEO_EXT:
        if s.lower().endswith(e):
            s = s[: -len(e)]
            break
    return s.lstrip("0") or "0"


def index_videos(split_dir):
    """Walk a split directory and map normalised clip id -> file path."""
    idx = {}
    dups = []
    if not split_dir.is_dir():
        return idx, dups
    for p in split_dir.rglob("*"):
        if p.suffix.lower() in VIDEO_EXT and p.is_file():
            k = norm_id(p.stem)
            if k in idx:
                dups.append((k, str(idx[k]), str(p)))
            else:
                idx[k] = p
    return idx, dups


def read_labels(csv_path):
    """Read a DAiSEE label CSV -> {normalised clip id: {col: int}}."""
    if not csv_path.is_file():
        return {}, f"missing label file: {csv_path}"
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        fields = {f.strip(): f for f in rdr.fieldnames or []}
        id_field = next((fields[f] for f in fields
                         if f.lower().replace(" ", "") in ("clipid", "clip_id", "id")), None)
        if id_field is None:
            return {}, f"no ClipID column in {csv_path}; saw {rdr.fieldnames}"
        missing = [c for c in LABEL_COLS if c not in fields]
        if missing:
            return {}, f"{csv_path} is missing columns {missing}; saw {rdr.fieldnames}"
        for row in rdr:
            k = norm_id(row[id_field])
            try:
                out[k] = {c: int(float(row[fields[c]])) for c in LABEL_COLS}
            except (ValueError, TypeError):
                continue
    return out, None


def sample_frames(video_path):
    """Uniformly sample SEQ_LEN frames -> uint8 (SEQ_LEN, H, W, 3) RGB.
    Returns None if the video cannot be read."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    picked = []
    if total > 0:
        for i in np.linspace(0, total - 1, SEQ_LEN).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                picked = []
                break
            picked.append(fr)

    if not picked:                      # header lied, or seeking failed — read it through
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        allf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr)
        cap.release()
        if not allf:
            return None
        picked = [allf[i] for i in np.linspace(0, len(allf) - 1, SEQ_LEN).astype(int)]
    else:
        cap.release()

    out = np.empty((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for j, fr in enumerate(picked):
        fr = cv2.resize(fr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        out[j] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)   # cv2 reads BGR; ImageNet wants RGB
    return out


def _worker(args):
    i, clip_id, path = args
    arr = sample_frames(Path(path))
    return i, clip_id, path, arr


# ----------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------
def inspect():
    print("=" * 68)
    print("PATHS")
    print("=" * 68)
    for name, p in [("ROOT", ROOT), ("VIDEO_DIR", VIDEO_DIR), ("LABEL_DIR", LABEL_DIR)]:
        print(f"  {name:10} {p}   {'OK' if p.exists() else '>>> NOT FOUND <<<'}")
    if VIDEO_DIR.is_dir():
        print(f"  subdirs of VIDEO_DIR: {sorted(d.name for d in VIDEO_DIR.iterdir() if d.is_dir())}")
    if LABEL_DIR.is_dir():
        print(f"  files in LABEL_DIR:   {sorted(f.name for f in LABEL_DIR.iterdir() if f.is_file())}")

    official = {"Train": 5358, "Validation": 1429, "Test": 1784}
    grand_missing = 0

    for dirname, labelfile in [("Train", "TrainLabels.csv"),
                               ("Validation", "ValidationLabels.csv"),
                               ("Test", "TestLabels.csv")]:
        print()
        print("=" * 68)
        print(f"SPLIT: {dirname}")
        print("=" * 68)

        vids, dups = index_videos(VIDEO_DIR / dirname)
        labels, err = read_labels(LABEL_DIR / labelfile)

        print(f"  videos found on disk : {len(vids)}")
        print(f"  label rows in CSV    : {len(labels)}" + (f"   [{err}]" if err else ""))
        print(f"  DAiSEE official count: {official.get(dirname, '?')}")
        if dups:
            print(f"  >>> {len(dups)} duplicate clip ids, e.g. {dups[:2]}")

        if labels:
            have = set(labels) & set(vids)
            no_video = sorted(set(labels) - set(vids))
            no_label = sorted(set(vids) - set(labels))
            print(f"  usable (label+video) : {len(have)}")
            if no_video:
                grand_missing += len(no_video)
                print(f"  >>> {len(no_video)} labelled clips have NO VIDEO on disk")
                print(f"      first 10: {no_video[:10]}")
            if no_label:
                print(f"  >>> {len(no_label)} videos have no label row (will be skipped)")

            y = np.array([labels[k]["Confusion"] for k in sorted(have)])
            if len(y):
                cnt = np.bincount(y, minlength=4)
                print(f"  Confusion distribution: {cnt.tolist()}   n={len(y)}")
                print(f"  majority baseline     : {cnt.max() / len(y):.4f}")

        # decode probe on a few real files
        probe = list(vids.values())[:3]
        for p in probe:
            arr = sample_frames(p)
            if arr is None:
                print(f"  >>> DECODE FAILED: {p}")
            else:
                print(f"  decode OK {p.name}: shape={arr.shape} dtype={arr.dtype} "
                      f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")

    print()
    print("=" * 68)
    print("SANITY CHECK AGAINST YOUR EXISTING FLOAT32 MMAPS")
    print("=" * 68)
    for name, shape in [("train_X.mmap", (4885, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
                        ("val_X.mmap",   (1429, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))]:
        f = ROOT / name
        if not f.is_file():
            print(f"  {name}: not found")
            continue
        X = np.memmap(f, dtype="float32", mode="r", shape=shape)
        s = np.array(X[0])
        print(f"  {name}: min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f}")
        print(f"      -> old data is {'[0,1] (needed Rescaling(255.0))' if s.max() <= 1.01 else '[0,255]'}")
        # channel means tell you whether the old build wrote RGB or BGR
        print(f"      channel means (c0,c1,c2): "
              f"{s[..., 0].mean():.4f}, {s[..., 1].mean():.4f}, {s[..., 2].mean():.4f}")

    print()
    print(f"TOTAL labelled clips with no video on disk: {grand_missing}")
    print("If that number is > 0, recover those clips BEFORE building.")
    print()
    print("REMINDER: the new mmaps are uint8 [0,255].")
    print("          Remove Rescaling(255.0) from the model when you switch.")


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------
def build(split, workers):
    if split not in SPLITS:
        sys.exit(f"unknown split '{split}'; choose from {list(SPLITS)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # gather (clip_id, path, labeldict) across the one or two source dirs
    items = []
    for dirname, labelfile in SPLITS[split]:
        vids, _ = index_videos(VIDEO_DIR / dirname)
        labels, err = read_labels(LABEL_DIR / labelfile)
        if err:
            sys.exit(err)
        for k in sorted(set(labels) & set(vids)):
            items.append((k, vids[k], labels[k]))

    if not items:
        sys.exit(f"nothing to build for split '{split}' — run --inspect first")

    n = len(items)
    x_path = OUT_DIR / f"{split}_X_u8.mmap"
    print(f"[{split}] {n} clips -> {x_path}")
    print(f"[{split}] {n * SEQ_LEN * IMG_SIZE * IMG_SIZE * 3 / 1e9:.2f} GB as uint8")

    X = np.memmap(x_path, dtype="uint8", mode="w+",
                  shape=(n, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))

    tasks = [(i, k, str(p)) for i, (k, p, _) in enumerate(items)]
    ok_rows, failed = [], []

    with Pool(processes=workers) as pool:
        for done, (i, clip_id, path, arr) in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=8), start=1):
            if arr is None:
                failed.append((i, clip_id, path))
            else:
                X[i] = arr
                ok_rows.append(i)
            if done % 200 == 0 or done == n:
                print(f"    {done}/{n}  ({len(failed)} failed)", flush=True)

    X.flush()
    del X

    keep = np.array(sorted(ok_rows), dtype=np.int64)
    if len(keep) < n:
        # compact so there are no all-zero rows left in the tensor
        print(f"[{split}] compacting: dropping {n - len(keep)} unreadable clips")
        src = np.memmap(x_path, dtype="uint8", mode="r",
                        shape=(n, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
        tmp = OUT_DIR / f"{split}_X_u8.tmp"
        dst = np.memmap(tmp, dtype="uint8", mode="w+",
                        shape=(len(keep), SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
        for new_i, old_i in enumerate(keep):
            dst[new_i] = src[old_i]
        dst.flush()
        del src, dst
        os.replace(tmp, x_path)

    kept_items = [items[i] for i in keep]

    np.savez(OUT_DIR / f"{split}_labels4.npz",
             **{c: np.array([lab[c] for _, _, lab in kept_items], dtype=np.int32)
                for c in LABEL_COLS})

    with open(OUT_DIR / f"{split}_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "ClipID", "path"] + LABEL_COLS)
        for r, (k, p, lab) in enumerate(kept_items):
            w.writerow([r, k, str(p)] + [lab[c] for c in LABEL_COLS])

    if failed:
        with open(OUT_DIR / f"{split}_failed.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ClipID", "path"])
            for _, k, p in failed:
                w.writerow([k, p])

    y = np.array([lab["Confusion"] for _, _, lab in kept_items])
    cnt = np.bincount(y, minlength=4)
    print()
    print(f"[{split}] DONE — {len(kept_items)} clips written")
    print(f"[{split}] SHAPE = ({len(kept_items)}, {SEQ_LEN}, {IMG_SIZE}, {IMG_SIZE}, 3)  dtype=uint8")
    print(f"[{split}] Confusion distribution: {cnt.tolist()}")
    print(f"[{split}] majority baseline     : {cnt.max() / len(y):.4f}")
    if failed:
        print(f"[{split}] {len(failed)} clips failed to decode, see {split}_failed.csv")
    print()
    print("Write that SHAPE down — you need it for np.memmap() in the training script.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="scan and report only; build nothing")
    ap.add_argument("--split", choices=list(SPLITS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    if a.inspect or not a.split:
        inspect()
    else:
        build(a.split, a.workers)
