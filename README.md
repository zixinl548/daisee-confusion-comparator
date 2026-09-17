# DAiSEE Confusion Comparator

EfficientNetB0 + BiGRU pipeline for four-class Confusion classification on
DAiSEE. This is the **comparator** workflow: it reproduces an existing
clip-level classification approach so that a new method can be compared
against it under identical conditions.

Maintainer: Nancy Li (zixinl) · Advisor: Xianming Tan · UNC Chapel Hill

---

## 1. What this pipeline does

```
10-second clip  ->  8 frames @ 224x224  ->  EfficientNetB0 (per frame)
                ->  Dense 256  ->  BiGRU 128  ->  Dense 128  ->  Softmax 4
```

One label per clip: Confusion rated Very Low / Low / High / Very High.

Frames are stored as **uint8 in [0, 255]**, cast to float32 at batch time.
`keras.applications.EfficientNetB0` performs its own `Rescaling(1/255)` and
ImageNet normalisation internally, so [0, 255] is the correct input range.

> **Do not add a Rescaling layer to the model.**
> An earlier version of this pipeline stored frames as float32 in [0, 1].
> EfficientNetB0 then divided by 255 a second time, compressing the input
> range from roughly [-2.12, +2.25] to [-2.118, -2.101] — a 255x collapse
> that turned every frame into a near-uniform grey field. This silently
> defeated every training run for months without raising an error.
> If you change how frames are stored, verify the value range first.

---

## 2. Environment

Tested on UNC Longleaf.

```bash
module load python/3.12.4
module load cuda/12.6
source .venv/bin/activate

pip install tensorflow opencv-python-headless numpy scikit-learn
```

Use `opencv-python-headless`, not `opencv-python`: the cluster has no
display and the full build drags in GUI dependencies.

---

## 3. Data layout

Download DAiSEE from
<https://people.iith.ac.in/vineethnb/resources/daisee/index.html>
and arrange it as:

```
Emotion/
├── DataSet/
│   ├── Train/<subject>/<clip>/<clip>.avi
│   ├── Validation/...
│   └── Test/...
├── Labels/
│   ├── TrainLabels.csv
│   ├── ValidationLabels.csv
│   └── TestLabels.csv
├── build_mmap.py
├── train_e2e.py
├── run_build.sh
└── run_e2e_dev.sh
```

Adjust `ROOT`, `VIDEO_DIR`, `LABEL_DIR` at the top of `build_mmap.py` if
your paths differ.

> **Do not keep the dataset inside iCloud Drive, Dropbox, or any synced
> folder.** macOS evicts infrequently used files to the cloud and leaves
> only placeholders on disk. `rsync` cannot transfer placeholders and
> reports only a generic `error 23`. This is how 473 training clips went
> missing from the first build, undetected until the clip count was
> checked against the official split size.

---

## 4. Step one: inspect before building

```bash
python build_mmap.py --inspect
```

Writes nothing. Reports, per split: videos found on disk, label rows in the
CSV, how many are usable, which labelled clips have no video, the Confusion
distribution, the majority baseline, and a decode probe on three real files.

Read the output before continuing. In particular, compare the usable count
against the official DAiSEE split sizes (5,358 / 1,429 / 1,784). A shortfall
means missing files, not a bug in the script.

---

## 5. Step two: build the tensors

```bash
sbatch run_build.sh
```

Builds `val`, `test`, and `trainval` as uint8 memory-maps. CPU work only, so
it runs on the `general` partition and does not consume GPU allocation.
Roughly one to two hours for about 10,000 clips at 8 workers.

Each split produces three files in `tensors_u8/`:

| File | Contents |
|---|---|
| `<split>_X_u8.mmap` | uint8, shape `(N, 8, 224, 224, 3)` |
| `<split>_labels4.npz` | Boredom / Engagement / Confusion / Frustration, int32 |
| `<split>_manifest.csv` | row index -> ClipID -> source path |

**Keep the manifests.** They are what makes a run reproducible, and they are
how `train_e2e.py` tells Train rows from Validation rows inside `trainval`.

A `<split>_failed.csv` appears only if some clips could not be decoded.

There is no separate `train` mmap by design. `trainval` contains Train rows
followed by Validation rows, and the manifest records which is which, so dev
mode slices it rather than storing a second copy.

---

## 6. Step three: train

**Development runs** — train on Train rows, evaluate on Validation rows:

```bash
sbatch run_e2e_dev.sh
# or interactively:
python train_e2e.py --mode dev --seed 42
```

**Final run** — train on all of trainval, evaluate on Test:

```bash
python train_e2e.py --mode final --seed 42
```

Tuning options:

| Flag | Default | Use |
|---|---|---|
| `--seed` | 42 | Vary for stability estimates |
| `--epochs` | 12 | |
| `--lr` | 2e-5 | |
| `--oversample-cap N` | match class 1 | Lower it to reduce repetition of rare classes |
| `--augment` | off | Horizontal flip + brightness jitter on training batches |

Every run appends one row to `results.csv` with its configuration and
metrics.

### The protocol matters

Do all tuning in `--mode dev`. Run `--mode final` **once**, after the
configuration is frozen. Every decision made while looking at a split biases
the estimate from that split; Test is the only clean estimate available, and
it is spent the first time you optimise against it.

Published DAiSEE results follow the convention of training on train plus
validation and reporting on the test split, which is what `--mode final`
does. Numbers from `--mode dev` are not comparable to published benchmarks.

---

## 7. Two design decisions worth understanding

### Checkpoints are selected on balanced accuracy, not validation loss

`BalancedAcc` must be **first** in the callback list. It computes
`val_balanced_acc` and writes it into `logs`; `ModelCheckpoint`,
`EarlyStopping`, and `ReduceLROnPlateau` all read that key and will raise a
missing-key error if the ordering is changed.

This is not cosmetic. An earlier run monitored `val_loss` and restored a
checkpoint scoring **0.2975** balanced accuracy, discarding an epoch-2
checkpoint that scored **0.3471**. The lowest-loss epoch is not the
best-classifying epoch.

### Seeds are fixed, including the oversampler

`oversample_indices` draws from a seeded `np.random.default_rng`. Without
this, resampling alone moved a result from 0.44 to 0.38 across runs. A
comparator that cannot be reproduced is not usable as a comparator.

Report the mean and standard deviation across at least three seeds, five
preferably:

```bash
for s in 42 43 44 45 46; do
  python train_e2e.py --mode dev --seed $s
done
```

---

## 8. Reference numbers

Measured on this build. Counts reflect clips actually present on disk and
successfully decoded, which is below the official split sizes.

| Split | n | Confusion `[VL, L, H, VH]` | Majority baseline |
|---|---:|---|---:|
| Validation | 1,429 | `[942, 322, 153, 12]` | **0.6592** |
| Test | 1,728 | `[1167, 410, 130, 21]` | **0.6753** |
| trainval | 6,314 | `[4260, 1435, 550, 69]` | 0.6747 |

Published DAiSEE benchmark: **72.3%** for Confusion (LRCN), 57.9% for
Engagement.

### Known data gaps

- 473 Train clips absent from disk, concentrated in nine subject folders
  (IDs beginning 2056). Cause was iCloud eviction, see section 3.
- 36 Test clips labelled but absent from disk.
- 20 Test clips present but undecodable, listed in `tensors_u8/test_failed.csv`,
  heavily concentrated in subject 510009.

Missing data in DAiSEE clusters by subject rather than occurring at random.
This is worth stating explicitly in any limitations section.

---

## 9. Why raw accuracy is not enough

The majority baseline is roughly 0.67. A model that ignores its input
entirely and always answers "Very Low" achieves that score. Raw accuracy
therefore **rewards a model that has learned nothing**, and the published
72.3% sits only about 5 points above doing nothing at all.

Always report, together:

- **Balanced accuracy** — mean per-class recall; chance is 0.25
- **Macro F1** — precision and recall averaged across classes
- **Raw accuracy** alongside the majority baseline of the same split
- **The confusion matrix**, or at minimum per-class recall

A collapsed model shows raw accuracy exactly at the baseline with balanced
accuracy pinned near 0.25 and zero recall on three of four classes. A model
that is genuinely separating classes shows non-zero recall everywhere, even
when its raw accuracy is lower.

### The Very High class

Very High has 69 training clips in trainval, 12 in validation, and 21 in
test. One run recovered 1 of 12 validation clips from roughly 50 predictions,
which is noise rather than learning. Balanced accuracy averages over all four
classes, so this class depresses the metric by construction.

Whether to collapse Confusion to three classes or to a binary target is an
open design question, not something the pipeline decides.

---

## 10. Files

| File | Purpose |
|---|---|
| `build_mmap.py` | Inspect data integrity; build uint8 tensors |
| `run_build.sh` | Slurm wrapper for the build, CPU partition |
| `train_e2e.py` | Train and evaluate, dev or final mode |
| `run_e2e_dev.sh` | Slurm wrapper for a dev run, GPU partition |
| `results.csv` | Appended automatically, one row per run |
| `tensors_u8/` | Built tensors, labels, manifests |
| `archive/` | Superseded scripts, kept for provenance |
