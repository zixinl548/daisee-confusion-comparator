# DAiSEE Confusion Comparator

EfficientNetB0 + BiGRU pipeline for four-class Confusion classification on
DAiSEE. This is the **comparator** workflow: it reproduces an existing
clip-level classification approach under a correct, reproducible protocol so
that a new method can be compared against it on equal terms.

Maintainer: Nancy Li (zixinl) · Advisor: Xianming Tan · UNC Chapel Hill

---

## 1. Headline result

Trained on train plus validation (6,314 clips), evaluated on the DAiSEE test
split (1,728 clips), three random seeds.

| Metric | Mean ± SD | Reference |
|---|---|---|
| Accuracy | **0.636 ± 0.008** | majority baseline 0.675 |
| Balanced accuracy | **0.311 ± 0.009** | chance 0.250 |
| Macro F1 | **0.301 ± 0.015** | |

Raw accuracy sits **0.039 below** the majority baseline. Balanced accuracy
sits 0.061 above chance.

**The model learns whether a student is confused, not how confused.** Of the
four levels, only the lowest two are separated reliably. See section 9.

This is a usable comparator. It is not a successful replication of the
published 72.3%.

---

## 2. What this pipeline does

```
10-second clip  ->  8 frames @ 224x224  ->  EfficientNetB0 (per frame)
                ->  Dense 256  ->  BiGRU 128  ->  Dense 128  ->  Softmax 4
```

One label per clip: Confusion rated Very Low / Low / High / Very High.

Frames are stored as **uint8 in [0, 255]**, cast to float32 at batch time.
`keras.applications.EfficientNetB0` performs its own `Rescaling(1/255)` and
ImageNet normalisation internally, so [0, 255] is the correct input range.

> **Do not add a Rescaling layer to the model.**
> An earlier version stored frames as float32 in [0, 1]. EfficientNetB0 then
> divided by 255 a second time, compressing the input range from roughly
> [-2.12, +2.25] to [-2.118, -2.101] — a 255x collapse that turned every
> frame into a near-uniform grey field. This silently defeated every training
> run for months without raising an error. If you change how frames are
> stored, verify the value range first.

---

## 3. Environment

Tested on UNC Longleaf.

```bash
module load python/3.12.4
module load cuda/12.6
source .venv/bin/activate

pip install tensorflow opencv-python-headless numpy scikit-learn
```

Use `opencv-python-headless`, not `opencv-python`: the cluster has no display
and the full build drags in GUI dependencies.

---

## 4. Data layout

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

Adjust `ROOT`, `VIDEO_DIR`, `LABEL_DIR` at the top of `build_mmap.py` if your
paths differ.

> **Do not keep the dataset inside iCloud Drive, Dropbox, or any synced
> folder.** macOS evicts infrequently used files to the cloud and leaves only
> placeholders on disk. `rsync` cannot transfer placeholders and reports only
> a generic `error 23`. This is how 473 training clips went missing from the
> first build, undetected until the clip count was checked against the
> official split size.

---

## 5. Step one: inspect before building

```bash
python build_mmap.py --inspect
```

Writes nothing. Reports, per split: videos found on disk, label rows in the
CSV, how many are usable, which labelled clips have no video, the Confusion
distribution, the majority baseline, and a decode probe on three real files.

Read the output before continuing. Compare the usable count against the
official DAiSEE split sizes (5,358 / 1,429 / 1,784). A shortfall means
missing files, not a bug in the script.

---

## 6. Step two: build the tensors

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

**Keep the manifests.** They make a run reproducible, and they are how
`train_e2e.py` tells Train rows from Validation rows inside `trainval`.

A `<split>_failed.csv` appears only if some clips could not be decoded.

There is no separate `train` mmap by design. `trainval` contains Train rows
followed by Validation rows, and the manifest records which is which, so dev
mode slices it rather than storing a second copy.

---

## 7. Step three: train

**Development runs** — train on Train rows, evaluate on Validation rows:

```bash
sbatch run_e2e_dev.sh
python train_e2e.py --mode dev --seed 42 --epochs 3
```

**Final runs** — train on all of trainval, evaluate on Test:

```bash
python train_e2e.py --mode final --seed 42 --epochs 3
```

Tuning options:

| Flag | Default | Use |
|---|---|---|
| `--seed` | 42 | Vary for stability estimates |
| `--epochs` | 12 | 3 is enough; the optimum appears in the first few epochs |
| `--lr` | 2e-5 | |
| `--oversample-cap N` | match class 1 | Lower it to reduce repetition of rare classes |
| `--augment` | off | Horizontal flip + brightness jitter on training batches |

Every run appends one row to `results.csv` with its configuration and metrics.

### Why the protocol matters

Do all tuning in `--mode dev`. Run `--mode final` after the configuration is
frozen.

This is not a formality. Balanced accuracy measured on validation was
**0.357 ± 0.014**; the same configuration measured on test was
**0.311 ± 0.009**. All three seeds dropped, without exception. That 0.046 gap
is the optimistic bias accumulated from tuning against validation. Test is
the only clean estimate available, and it is spent the first time you
optimise against it.

Published DAiSEE results follow the convention of training on train plus
validation and reporting on the test split, which is what `--mode final`
does. Numbers from `--mode dev` are not comparable to published benchmarks.

### Runtime

About 30 minutes per epoch in dev mode (375–696 steps), about 38 minutes in
final mode. Three epochs plus evaluation is roughly two hours.

Training is **GPU-bound, not I/O-bound**. Converting the tensors from float32
to uint8 cut stored volume by 75% and left per-epoch time essentially
unchanged, which rules out storage bandwidth as the bottleneck. The cost is
the full backward pass through EfficientNetB0 on 8 frames per clip.

---

## 8. Two design decisions worth understanding

### Checkpoints are selected on balanced accuracy, not validation loss

`BalancedAcc` must be **first** in the callback list. It computes
`val_balanced_acc` and writes it into `logs`; `ModelCheckpoint`,
`EarlyStopping`, and `ReduceLROnPlateau` all read that key and will raise a
missing-key error if the ordering changes.

This is not cosmetic. An earlier run monitored `val_loss` and restored a
checkpoint scoring **0.2975** balanced accuracy, discarding an epoch-2
checkpoint that scored **0.3471**. Across seven epochs, validation loss fell
monotonically while balanced accuracy peaked at epoch 1 and then declined.
The lowest-loss epoch is not the best-classifying epoch.

### Seeds are fixed, including the oversampler

`oversample_indices` draws from a seeded `np.random.default_rng`. Without
this, resampling alone moved an earlier result from 0.44 to 0.38 across runs.

Seed 42 has been reproduced exactly across two separate jobs with different
epoch budgets, both giving 0.3643 balanced accuracy on validation.

**Report mean and standard deviation across at least three seeds.** Single
runs are not interpretable here: a configuration change that looked harmful
in one run (`--oversample-cap 600`, 0.3347 versus 0.3643) fell well inside
the seed-to-seed spread once three seeds were available. Differences under
roughly 0.03 in balanced accuracy cannot be distinguished from noise.

```bash
for s in 42 43 44; do
  python train_e2e.py --mode final --seed $s --epochs 3
done
```

---

## 9. Results in detail

### Per-seed, test split

| Seed | Accuracy | Balanced acc | Macro F1 |
|---|---:|---:|---:|
| 42 | 0.6360 | 0.3103 | 0.2922 |
| 43 | 0.6279 | 0.3018 | 0.2927 |
| 44 | 0.6435 | 0.3202 | 0.3187 |
| **Mean ± SD** | **0.636 ± 0.008** | **0.311 ± 0.009** | **0.301 ± 0.015** |

### Per-class behaviour across all three seeds

| Class | n | Recall | Precision | Verdict |
|---|---:|---|---|---|
| Very Low | 1167 | 0.87 / 0.84 / 0.86 | 0.71 / 0.72 / 0.72 | learned, stable |
| Low | 410 | 0.19 / 0.24 / 0.22 | 0.37 / 0.40 / 0.38 | learned, weak but stable |
| High | 130 | 0.04 / 0.03 / 0.10 | 0.29 / 0.06 / 0.20 | **not learned** |
| Very High | 21 | 0.14 / 0.10 / 0.10 | 0.04 / 0.04 / 0.06 | **noise** |

**The two lowest levels are learned; the two highest are not.**

The evidence is in the precision stability. Low holds precision at
0.37 / 0.40 / 0.38 across seeds, consistently above the 0.237 rate expected
from guessing, which indicates real and repeatable discriminative signal.
High swings from 0.06 to 0.29 across seeds — a five-fold range that does not
represent anything learned. Very High recovers 3, 2, and 2 of its 21 clips
from roughly 50 predictions each time.

**Sample size alone does not explain this.** Very High has only 21 test clips,
so its instability is unsurprising. But High has 130 clips and still returns
near-zero recall. The problem is not that the rare levels are rare; it is that
the boundary between adjacent middle levels is not recoverable from facial
signal.

### Errors are ordinal

In every seed, Low is confused overwhelmingly with Very Low (304 / 271 / 282
cases) and rarely with Very High (25 / 16 / 17). Adjacent-class confusion
dominates jump confusion, which indicates the model perceives an ordered
intensity axis but resolves it to roughly two levels rather than four.

### Interpretation

This is consistent with Barrett et al., *Emotional Expressions Reconsidered*,
which argues that facial movements map many-to-many onto affective categories
rather than one-to-one. The measured behaviour here is what that position
predicts: facial signal supports a coarse binary judgement about confusion
and does not support a four-level ordinal scale.

Whether to collapse Confusion to three classes or to a binary target is
therefore an open design question with empirical support behind it, not
merely a workaround for class imbalance.

---

## 10. Reference numbers

Measured on this build. Counts reflect clips present on disk and successfully
decoded, which is below the official split sizes.

| Split | n | Confusion `[VL, L, H, VH]` | Majority baseline |
|---|---:|---|---:|
| Validation | 1,429 | `[942, 322, 153, 12]` | 0.6592 |
| Test | 1,728 | `[1167, 410, 130, 21]` | **0.6753** |
| trainval | 6,314 | `[4260, 1435, 550, 69]` | 0.6747 |

Published DAiSEE benchmark: **72.3%** for Confusion (LRCN), 57.9% for
Engagement.

### Known data gaps

- 473 Train clips absent from disk, concentrated in nine subject folders
  (IDs beginning 2056). Cause was iCloud eviction, see section 4.
- 36 Test clips labelled but absent from disk.
- 20 Test clips present but undecodable, listed in
  `tensors_u8/test_failed.csv`, heavily concentrated in subject 510009.

Missing data in DAiSEE clusters by subject rather than occurring at random.
Worth stating explicitly in any limitations section.

---

## 11. Why raw accuracy is not enough

The majority baseline on test is 0.675. A model that ignores its input and
always answers "Very Low" achieves that score. Raw accuracy therefore
**rewards a model that has learned nothing**, and the published 72.3% sits
only about 5 points above doing nothing at all.

Raw accuracy is also the **noisier** metric. On validation across three
seeds, accuracy varied by ±0.043 while balanced accuracy varied by ±0.014.
Accuracy is highly sensitive to how many borderline cases the model assigns
to the majority class, and that tendency swings from run to run.

Always report, together:

- **Balanced accuracy** — mean per-class recall; chance is 0.25
- **Macro F1** — precision and recall averaged across classes
- **Raw accuracy** alongside the majority baseline of the same split
- **The confusion matrix**, or at minimum per-class recall and precision

A collapsed model shows raw accuracy exactly at the baseline with balanced
accuracy pinned near 0.25 and zero recall on three of four classes. A model
genuinely separating classes shows non-zero recall everywhere, even when its
raw accuracy is lower.

---

## 12. Files

| File | Purpose |
|---|---|
| `build_mmap.py` | Inspect data integrity; build uint8 tensors |
| `run_build.sh` | Slurm wrapper for the build, CPU partition |
| `train_e2e.py` | Train and evaluate, dev or final mode |
| `run_e2e_dev.sh` | Slurm wrapper for a dev run, GPU partition |
| `results.csv` | Appended automatically, one row per run |
| `tensors_u8/` | Built tensors, labels, manifests |
| `archive/` | Superseded scripts, kept for provenance |
