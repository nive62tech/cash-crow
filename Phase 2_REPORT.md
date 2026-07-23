# Smart Bin — Phase 2 Report: Waste Classification Integration

## 1. Overview

Phase 1 built a live detection pipeline: YOLOv8n (pre-trained on COCO)
locates objects in the webcam feed, MediaPipe Hand Landmarker locates
hands, and custom overlap logic decides whether a detected object is
"held." A display state machine then drew a red box around held
objects — but labeled every one of them with the generic word
"Object," since YOLO's 80 COCO classes have no concept of
plastic/paper/trash.

**Phase 2's goal:** replace that generic "Object" label with a real
prediction — Plastic, Paper, or Other — using a second, separate
model that looks only at the cropped held-object region.

This report documents the dataset, the training run, the classifier
wrapper, its integration into the live pipeline, and the debugging
process used to evaluate real-world accuracy.

---

## 2. Why a Second Model (Not Just YOLO)

YOLOv8n's 80 COCO classes (person, chair, cell phone, bottle, remote,
toothbrush, etc.) don't include material categories like "plastic" or
"paper." Asking YOLO to answer "what material is this" would require
retraining YOLO itself on a completely different label set, which is
unnecessary extra work for a task that doesn't need object
*localization* — only *classification of an already-known region*.

Instead, Phase 2 introduces a **second, independent model**:

- **YOLOv8n** still answers "where is an object in this frame."
- **MobileNetV3-Small** (new) answers "given just this cropped
  picture of one object, what material is it."

MobileNetV3-Small was chosen specifically for its small size and
speed, since it only ever processes a small cropped region (not the
full frame), keeping the eventual target of Raspberry Pi deployment
realistic.

---

## 3. Dataset

- **Source:** a Kaggle waste-classification dataset, sorted into 3
  flat folders under `data/raw/`.
- **Class sizes:**
  | Class | Images |
  |---|---|
  | Plastic | 921 |
  | Paper | 961 |
  | Other (food waste, glass, metal, vegetation, misc.) | 1,355 |

  Note the **class imbalance** — Other has roughly 40% more images
  than Plastic or Paper. This was flagged early as a risk for
  prediction bias (see Section 8).

- **Splitting:** `split_dataset.py` performs an 80/20 train/val split
  per class, producing `data/classification/train/<class>/` and
  `data/classification/val/<class>/` in `torchvision.ImageFolder`
  format.
  - **Train:** 2,590 images
  - **Val:** 647 images

---

## 4. Training (`train_classifier.py`)

- **Base model:** `mobilenet_v3_small`, pre-trained on ImageNet
  (transfer learning) via `MobileNet_V3_Small_Weights.DEFAULT`.
- **Head replacement:** only the final classification layer
  (`model.classifier[-1]`) is swapped from 1000 ImageNet classes down
  to 3 (Plastic / Paper / Other). The rest of the pre-trained
  backbone is reused as-is, so the model starts from general visual
  features rather than learning from scratch.
- **Input size:** 224×224, matching MobileNetV3's expected input.
- **Augmentation (train set only):** random horizontal flip, random
  rotation (±15°), color jitter (brightness/contrast/saturation ±0.2).
  Applied only to training data — validation images are left
  unaltered so val accuracy reflects real performance, not performance
  on artificially altered images.
- **Normalization:** standard ImageNet mean/std
  (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`), applied to
  both train and val sets.
- **Optimizer:** Adam, learning rate `0.001`.
- **Loss:** CrossEntropyLoss.
- **Batch size:** 8 (lowered from an initial 16 after hitting an
  out-of-memory crash on CPU).
- **Epochs:** up to 15, but only the checkpoint with the *best* val
  accuracy is saved — not necessarily the final epoch.
- **Checkpoint format:** a single file,
  `models/mobilenetv3_waste.pth`, saved as a dict:
  ```python
  { "model_state_dict": ..., "class_names": ["Other", "Paper", "Plastic"] }
  ```
  Storing `class_names` alongside the weights was a deliberate
  decision so that inference code never has to hardcode or guess the
  class order — it reads the order directly from the checkpoint,
  removing a whole category of "silent mislabeling" bugs.

### Training results

| Epoch | Val accuracy |
|---|---|
| 1 | 78.4% |
| 2 | 88.6% |
| 6 | **92.3% (best)** |

No instability was observed during training. Training was run in a
standalone PowerShell window (outside VS Code) after the VS Code
integrated terminal repeatedly froze under sustained CPU load.

**Important caveat (see Section 8):** this 92.3% reflects accuracy on
*held-out Kaggle images* — clean, centered, well-lit product-style
photos — not on live webcam captures. The two are not the same
distribution, and this gap became the central focus of live testing.

---

## 5. `classifier.py` — Inference Wrapper

A standalone module that:

1. **Loads the checkpoint once at import time** (not on every
   prediction) — the model and `class_names` list are loaded into
   module-level variables the first time `classifier.py` is imported,
   so repeated calls to `predict()` don't reload the model from disk.
2. **Rebuilds the exact same architecture** used in training
   (`mobilenet_v3_small` with the final layer swapped to `len(class_names)`
   outputs) before loading the saved weights into it — this has to
   match exactly, or `load_state_dict()` fails or silently mismatches.
3. **Applies the same preprocessing as training's validation
   transform** — `Resize((224,224))` → `ToTensor()` → `Normalize(...)`
   with identical mean/std — since any mismatch here (a common,
   easy-to-miss bug) would cause predictions that don't match what was
   measured during training/validation.
4. **Exposes one function:**
   ```python
   predict(cropped_image) -> (label: str, confidence: float)
   ```
   Accepts either an OpenCV BGR NumPy array (as produced by cropping a
   frame) or a PIL RGB image, and internally converts as needed.
5. **Applies a confidence threshold** (`CONFIDENCE_THRESHOLD = 0.6`):
   if the model's top prediction is below this confidence, `predict()`
   returns `"Uncertain"` instead of forcing a possibly-wrong label.
6. **Debug mode** (`DEBUG_SAVE_CROPS = True`): saves every crop passed
   to `predict()` into a `debug_crops/` folder, and prints the full
   3-class probability breakdown for every prediction (not just the
   winning class) — added specifically to diagnose *why* a prediction
   was wrong, not just *that* it was wrong.

**Standalone verification:** before touching the live pipeline,
`classifier.py` was tested in isolation against a real image from the
val set (`Plastic_...jpg`) and correctly predicted `Plastic` at 98.7%
confidence — confirming the architecture, checkpoint loading, and
preprocessing were all wired correctly before adding the complexity of
live video.

---

## 6. Integration into `detect.py`

Four changes were made to the Phase 1 pipeline:

1. **Cropping.** A new `crop_object()` helper crops the frame to a
   held object's bounding box, with ~10% padding on each side so the
   classifier isn't fed an overly tight crop that clips the object's
   edges.

2. **Classification trigger.** Inside `render_frame()`, whenever the
   display state is `"object"` **and** a *freshly detected* held
   object exists this frame (not the cached fallback box used during
   brief detection flicker), the highest-scoring held object
   (`roi_selector.py`'s existing `.score`) is cropped and passed to
   `classifier.predict()`.

3. **Prediction smoothing.** Raw per-frame predictions are pushed into
   a 5-frame rolling history (`DisplayState.prediction_history`, a
   `deque(maxlen=5)`) and majority-voted
   (`get_smoothed_prediction()`) before being displayed. This mirrors
   the same debouncing philosophy Phase 1 already used for the
   red/green/blue detection state itself — a single noisy frame can't
   flip the displayed label. History is cleared whenever the object
   state ends, so stale votes don't leak into the next held item.

4. **Label display.** `draw_held_objects()` now accepts a `label_text`
   parameter, so the red box shows e.g. `Plastic (94%)` instead of the
   static word "Object."

---

## 7. Debugging Process

Early live testing showed inconsistent results: real plastic items
sometimes read as `Uncertain` or `Other`; a pen was misread as
`Paper`; predictions occasionally flickered between classes across
frames.

To investigate without guessing, two debugging tools were added to
`classifier.py`:

- **`debug_crops/`** — every image actually fed to the model is saved
  to disk, so the crop quality itself can be visually inspected
  (centered vs. clipped, blurry vs. sharp, clean background vs.
  cluttered).
- **Full probability logging** — every prediction prints all 3 class
  probabilities (e.g. `Other=0.71, Paper=0.15, Plastic=0.14`), not
  just the winning label, to distinguish "confidently wrong" (a
  crop/data problem) from "genuinely torn between classes" (a
  threshold/training problem).

**Working hypothesis:** the most likely cause of live inaccuracy is a
**domain gap** between training data and live captures, rather than a
bug in the wiring. The Kaggle training images are clean, centered,
well-lit product-style photos on plain backgrounds. Live webcam crops
are, by contrast: partially occluded by fingers, off-center, often
motion-blurred, shot against cluttered real-world backgrounds, and
squeezed from a rectangular COCO bounding box into a square 224×224
input (potentially distorting aspect ratio). Any of these shifts an
image "out of distribution" relative to what the model was trained
on — independent of how well the model performed on its own held-out
validation set.

A secondary contributing factor: YOLO itself has no concept of "pen,"
"seal cap," "attar bottle," or "mouse" — it forces these into the
nearest COCO class it does know (`remote`, `cell phone`, `toothbrush`,
`bottle`). Since the classifier only ever sees the region *inside*
whatever box YOLO drew, a loose or oddly-shaped COCO box can itself
degrade the crop quality feeding into classification — a separate
issue from the classifier's own accuracy.

**Status at time of writing:** debug tooling is in place and
confirmed working (crops saving to `debug_crops/`, probability
breakdowns printing per-frame). Systematic live testing across
Plastic/Paper/Other with this tooling active is the next step before
drawing final conclusions on root cause and fix (see Section 9).

---

## 8. Class Imbalance Check

Because "Other" (1,355 images) is over-represented relative to
Plastic (921) and Paper (961) in training, there was a standing
concern that the model would be statistically biased toward
predicting "Other" regardless of the true class — a common failure
mode when one class dominates a training set.

Live testing did show `Other` appearing frequently, including for at
least one item (a pen) that should have read as a different class.
Whether this reflects genuine class-imbalance bias, the domain-gap
issue described in Section 7, or both, has not yet been conclusively
separated — that determination requires the systematic per-class
testing with debug logging described in Section 9.

---

## 9. Current Status & Next Steps

**Completed:**
- Dataset split and organized (`split_dataset.py`)
- Model trained to 92.3% val accuracy (`train_classifier.py`)
- Inference wrapper built and unit-verified (`classifier.py`)
- Full integration into the live pipeline (`detect.py`), including
  cropping, confidence thresholding, and frame-smoothing
- Debug tooling added (crop saving + full probability logging) to
  support root-cause diagnosis of live accuracy issues
- Repository cleaned: dataset images, model checkpoint, and the
  auto-downloaded hand-landmark model were removed from git tracking
  and purged from git history (repo size reduced from ~469 MB to
  ~5.6 MB), since none of these belong in version control — the
  dataset and checkpoint are regenerable/retrainable artifacts, and
  the hand-landmark model is already auto-downloaded at runtime by
  `detect.py`.

**Remaining (Phase 2 close-out):**
1. Run a systematic live test pass across all 3 classes with debug
   logging active, and review the saved crops + probability
   breakdowns to conclusively identify whether live inaccuracy stems
   from crop quality, domain gap, class imbalance, or a combination.
2. Based on findings, apply the appropriate fix — likely candidates
   include: adjusting `CONFIDENCE_THRESHOLD`, improving the crop
   padding/aspect-ratio handling, or (most likely to meaningfully
   close the gap) augmenting the training set with more realistic,
   cluttered, hand-held-style photos rather than relying solely on
   clean product shots.
3. Re-test live and confirm improved accuracy across all 3 classes.
4. Final documentation update and clean commit/push to close out
   Phase 2.

---

## 10. File Summary

| File | Purpose |
|---|---|
| `split_dataset.py` | 80/20 train/val split of raw Kaggle images |
| `train_classifier.py` | Trains MobileNetV3-Small, saves best checkpoint |
| `models/mobilenetv3_waste.pth` | Trained weights + class name order (not committed to git — see Section 9) |
| `classifier.py` | Loads checkpoint, exposes `predict(cropped_image)` |
| `detect.py` | Phase 1 detection pipeline + Phase 2 classification integration |
| `debug_crops/` | Saved model input crops for visual debugging (local only, gitignored) |