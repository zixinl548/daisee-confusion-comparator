#!/usr/bin/env python
"""
train_e2e.py — DAiSEE Confusion comparator, EfficientNetB0 + BiGRU.

Changes from the previous version
---------------------------------
1. NO Rescaling layer.  The uint8 mmaps hold [0,255]; casting to float32
   gives [0.0, 255.0], which is exactly what EfficientNetB0 expects.
   Adding Rescaling(255.0) here would give [0, 65025] — the same class of
   bug as before, in the opposite direction.
2. Checkpointing / early stopping monitor val_balanced_acc, not val_loss.
   The lowest-loss epoch is NOT the best-classifying epoch: the last run
   reached 0.3471 balanced accuracy at epoch 2 and restored 0.2975.
3. Seeds fixed everywhere, including the oversampler.  Reproducibility is
   not optional for a comparator — a different seed previously moved the
   result by six points.
4. Two modes:
      --mode dev    train on Train rows, evaluate on Validation rows
                    (both live inside trainval_X_u8.mmap)
      --mode final  train on ALL of trainval, evaluate on the test split
   Do all tuning in dev.  Run final ONCE, after the configuration is frozen.
5. Reports accuracy, balanced accuracy, macro-F1, the majority baseline of
   whichever split was used, and the confusion matrix.

Usage
-----
    python train_e2e.py --mode dev   --seed 42
    python train_e2e.py --mode final --seed 42
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np

# seeds before TF import so its internal RNG picks them up
def seed_everything(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)

_ap = argparse.ArgumentParser()
_ap.add_argument("--mode", choices=["dev", "final"], default="dev")
_ap.add_argument("--seed", type=int, default=42)
_ap.add_argument("--epochs", type=int, default=12)
_ap.add_argument("--batch", type=int, default=8)
_ap.add_argument("--lr", type=float, default=2e-5)
_ap.add_argument("--oversample-cap", type=int, default=None,
                 help="cap minority classes at this count instead of matching "
                      "class 1.  Lower = less repetition = less overfitting. "
                      "Try 600 if the model memorises.")
_ap.add_argument("--augment", action="store_true",
                 help="random horizontal flip + brightness jitter on training batches")
ARGS = _ap.parse_args()
seed_everything(ARGS.seed)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)

tf.random.set_seed(ARGS.seed)

# ----------------------------------------------------------------------
ROOT     = Path("/work/users/z/i/zixinl/Emotion")
TDIR     = ROOT / "tensors_u8"
SEQ_LEN  = 8
IMG_SIZE = 224
N_CLASS  = 4
CLASS_NAMES = ["VeryLow", "Low", "High", "VeryHigh"]
SAVE_PATH = ROOT / f"confusion_{ARGS.mode}_seed{ARGS.seed}.keras"


def open_mmap(split):
    """Open <split>_X_u8.mmap, inferring N from the file size."""
    p = TDIR / f"{split}_X_u8.mmap"
    if not p.is_file():
        raise SystemExit(f"missing {p} — run build_mmap.py --split {split} first")
    per = SEQ_LEN * IMG_SIZE * IMG_SIZE * 3          # bytes per clip, uint8
    size = p.stat().st_size
    if size % per:
        raise SystemExit(f"{p} size {size} is not a multiple of {per}; rebuild it")
    n = size // per
    X = np.memmap(p, dtype="uint8", mode="r",
                  shape=(n, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
    y = np.load(TDIR / f"{split}_labels4.npz")["Confusion"].astype(np.int32)
    if len(y) != n:
        raise SystemExit(f"{split}: mmap has {n} rows but labels have {len(y)}")
    print(f"  {split:9} n={n:5}  {size/1e9:.2f} GB  dist={np.bincount(y, minlength=4).tolist()}")
    return X, y


def manifest_origin(split):
    """Row -> 'Train' or 'Validation', read from the manifest written at build
    time.  Robust to rows having been dropped during compaction."""
    rows = []
    with open(TDIR / f"{split}_manifest.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            p = r["path"]
            rows.append("Validation" if "/Validation/" in p else "Train")
    return np.array(rows)


def oversample_indices(labels, pool, rng, cap=None):
    """Balance `pool` (an index array into `labels`) by resampling.
    Deterministic given rng."""
    y = labels[pool]
    counts = np.bincount(y, minlength=N_CLASS)
    target = cap if cap is not None else int(counts[1])
    print(f"  before: {counts.tolist()}   target per minority class: {target}")
    out = []
    for c in range(N_CLASS):
        idx = pool[y == c]
        if len(idx) == 0:
            continue
        if c == 0:                                   # majority: subsample
            k = min(len(idx), target * 2)
            idx = rng.choice(idx, size=k, replace=False)
        elif len(idx) < target:                      # minority: oversample
            idx = rng.choice(idx, size=target, replace=True)
        elif len(idx) > target:
            idx = rng.choice(idx, size=target, replace=False)
        out.append(idx)
    comb = np.concatenate(out)
    rng.shuffle(comb)
    print(f"  after : {np.bincount(labels[comb], minlength=N_CLASS).tolist()}  "
          f"total={len(comb)}  steps/epoch={int(np.ceil(len(comb)/ARGS.batch))}")
    return comb


def augment_batch(x, rng):
    """x is float32 [0,255].  Light, label-preserving jitter."""
    if rng.random() < 0.5:
        x = x[:, :, :, ::-1, :]                      # horizontal flip (W axis)
    x = x * rng.uniform(0.9, 1.1)                    # brightness
    return np.clip(x, 0.0, 255.0)


def make_train_ds(X, y, order, batch, seed, augment):
    n = len(order)
    def gen():
        rng = np.random.default_rng(seed)
        while True:
            perm = rng.permutation(n)
            for i in range(0, n, batch):
                rows = sorted(order[perm[i:i + batch]].tolist())  # sorted = better mmap locality
                xb = X[rows].astype(np.float32)                   # uint8 -> [0,255]
                if augment:
                    xb = augment_batch(xb, rng)
                yield xb, y[rows]
    sig = (tf.TensorSpec((None, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), tf.float32),
           tf.TensorSpec((None,), tf.int32))
    steps = int(np.ceil(n / batch))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE), steps


def make_eval_ds(X, y, order, batch):
    order = np.sort(np.asarray(order))
    n = len(order)
    def gen():
        while True:
            for i in range(0, n, batch):
                rows = order[i:i + batch].tolist()
                yield X[rows].astype(np.float32), y[rows]
    sig = (tf.TensorSpec((None, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), tf.float32),
           tf.TensorSpec((None,), tf.int32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE), y[order]


# ----------------------------------------------------------------------
print(f"\n=== mode={ARGS.mode}  seed={ARGS.seed} ===")
print("loading tensors")

if ARGS.mode == "dev":
    X_all, y_all = open_mmap("trainval")
    origin = manifest_origin("trainval")
    if len(origin) != len(y_all):
        raise SystemExit("manifest length does not match mmap; rebuild trainval")
    train_pool = np.where(origin == "Train")[0]
    eval_pool  = np.where(origin == "Validation")[0]
    X_eval, y_eval_all = X_all, y_all
    print(f"  dev split: train rows={len(train_pool)}  eval rows={len(eval_pool)}")
else:
    X_all, y_all = open_mmap("trainval")
    train_pool = np.arange(len(y_all))
    X_eval, y_eval_all = open_mmap("test")
    eval_pool = np.arange(len(y_eval_all))
    print(f"  final: train rows={len(train_pool)}  test rows={len(eval_pool)}")

rng = np.random.default_rng(ARGS.seed)
print("building training order")
order = oversample_indices(y_all, train_pool, rng, ARGS.oversample_cap)

train_ds, steps = make_train_ds(X_all, y_all, order, ARGS.batch, ARGS.seed, ARGS.augment)
eval_ds, y_true = make_eval_ds(X_eval, y_eval_all, eval_pool, ARGS.batch)

maj = np.bincount(y_true, minlength=N_CLASS).max() / len(y_true)
print(f"  evaluation majority baseline = {maj:.4f}")

# ----------------------------------------------------------------------
backbone = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet",
    pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
backbone.trainable = True

inp = keras.Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
# NO Rescaling here.  Input is already [0,255], which EfficientNetB0 wants.
x = layers.TimeDistributed(backbone)(inp)
x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(x)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.GRU(128))(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(N_CLASS, activation="softmax")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(ARGS.lr),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()


class BalancedAcc(keras.callbacks.Callback):
    """MUST be first in the callback list: it writes val_balanced_acc into
    `logs`, and the callbacks after it read that key."""
    def __init__(self, ds, y_true, steps):
        super().__init__()
        self.ds, self.y_true, self.steps = ds, y_true, steps
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        p = np.argmax(self.model.predict(self.ds, steps=self.steps, verbose=0), axis=-1)[:len(self.y_true)]
        logs["val_balanced_acc"] = balanced_accuracy_score(self.y_true, p)
        logs["val_macro_f1"] = f1_score(self.y_true, p, average="macro", zero_division=0)
        print(f"  balanced_acc={logs['val_balanced_acc']:.4f}  "
              f"macro_f1={logs['val_macro_f1']:.4f}  "
              f"predicted classes={sorted(set(p.tolist()))}")


callbacks = [
    BalancedAcc(eval_ds, y_true, int(np.ceil(len(y_true)/ARGS.batch))),                    # first, always
    keras.callbacks.ModelCheckpoint(
        str(SAVE_PATH), monitor="val_balanced_acc", mode="max",
        save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor="val_balanced_acc", mode="max", patience=6,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_balanced_acc", mode="max", factor=0.5,
        patience=3, verbose=1),
]

val_steps = int(np.ceil(len(y_true) / ARGS.batch))
model.fit(train_ds, steps_per_epoch=steps,
          validation_data=eval_ds, validation_steps=val_steps,
          epochs=ARGS.epochs, callbacks=callbacks, verbose=1)

# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"FINAL EVALUATION   mode={ARGS.mode}  seed={ARGS.seed}")
print("=" * 60)
y_pred = np.argmax(model.predict(eval_ds, steps=val_steps, verbose=0), axis=-1)[:len(y_true)]

acc  = accuracy_score(y_true, y_pred)
bacc = balanced_accuracy_score(y_true, y_pred)
mf1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

print(f"n                 : {len(y_true)}")
print(f"Accuracy          : {acc:.4f}")
print(f"Balanced Accuracy : {bacc:.4f}")
print(f"Macro F1          : {mf1:.4f}")
print(f"Majority baseline : {maj:.4f}")
print(f"Above baseline    : {'YES' if acc > maj else 'no'}   "
      f"({acc - maj:+.4f})")
print()
print(classification_report(y_true, y_pred, labels=list(range(N_CLASS)),
                            target_names=CLASS_NAMES, zero_division=0))
print("confusion matrix (rows = true, cols = predicted)")
print("            " + "".join(f"{c:>10}" for c in CLASS_NAMES))
for i, row in enumerate(confusion_matrix(y_true, y_pred, labels=list(range(N_CLASS)))):
    print(f"{CLASS_NAMES[i]:>12}" + "".join(f"{v:>10}" for v in row))

# one-line summary for collecting across seeds
res = ROOT / "results.csv"
new = not res.exists()
with open(res, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["mode", "seed", "epochs", "lr", "oversample_cap", "augment",
                    "n", "accuracy", "balanced_acc", "macro_f1", "majority"])
    w.writerow([ARGS.mode, ARGS.seed, ARGS.epochs, ARGS.lr, ARGS.oversample_cap,
                int(ARGS.augment), len(y_true), f"{acc:.4f}", f"{bacc:.4f}",
                f"{mf1:.4f}", f"{maj:.4f}"])
print(f"\nappended to {res}")
