# detect.py changes needed for the 10-class scheme

I don't have your actual current `detect.py`, so this is NOT a full-file
rewrite -- it's the specific pieces that need to change, for you to paste
into the right spots yourself. Rewriting the whole file blind (with YOLO +
MediaPipe wiring I haven't seen) would be guessing, not helping.

## 1. Add a class -> physical compartment mapping

Add this near your other config constants (near CONFIDENCE_THRESHOLD,
HAND_REGION_PADDING_RATIO, etc). This is the ONE place that decides what
happens physically for each of the 10 classes -- update it once your team
confirms the real compartment count/design.

```python
# Maps classifier output -> physical bin compartment.
# ASSUMPTION: bin currently only has compartments for a subset of these.
# Update the values once hardware compartments are finalized.
CLASS_TO_COMPARTMENT = {
    "Plastic":        "compartment_1",
    "Paper":          "compartment_2",
    "Glass":          "compartment_3",
    "Metal":          "compartment_4",
    "Organic":        "compartment_5",
    "Cloth":          "compartment_6",   # NEW -- confirm hardware exists
    "Shoes":          "compartment_6",   # sharing with Cloth unless you want separate
    "Hazard":         "compartment_7",   # NEW -- confirm hardware exists
    "Non-Recyclable": "compartment_8",   # NEW -- confirm hardware exists
    "Uncertain":      None,              # no compartment -- don't act
}
```

If your bin genuinely only has fewer physical compartments right now, this
is the layer where you collapse classes down -- e.g. if Cloth/Shoes/Hazard/
Non-Recyclable all have to share one physical "misc" bin for now, just point
all four at the same compartment string here. The classifier itself still
reports the correct 10-way label; only the physical routing gets collapsed.

## 2. Where classification currently runs on a held object

Find the block that calls `classifier.predict(...)` on a freshly detected
held object (per the original doc: "Classification only runs on freshly
detected held objects, not cached fallback boxes"). No changes needed to
the call itself -- classifier.py's predict() interface is unchanged:

```python
label, confidence = classifier.predict(cropped_image)
```

`label` will now be one of the 10 class names or "Uncertain" -- same shape
as before, just more possible values.

## 3. Prediction smoothing (rolling deque + majority vote)

No changes needed to the mechanism itself. Just double check
PREDICTION_HISTORY_SIZE is still sensible -- with 10 classes instead of 3,
a short history window may cause more flip-flopping between visually
similar classes (e.g. Glass vs Plastic, Cloth vs Shoes). If you see jitter
in testing, try increasing PREDICTION_HISTORY_SIZE from whatever it's
currently set to (doc says it was tuned between 2 and 5).

## 4. draw_held_objects() / label display

Wherever the box label text is built (doc describes this as producing
something like "Plastic (94%)"), that logic doesn't need to change --
`label` and `confidence` are still a string and a float. Just make sure
whatever compartment-triggering logic exists downstream now looks up
`CLASS_TO_COMPARTMENT[label]` instead of assuming one of the old 3 classes.

## 5. One thing to test deliberately

"Uncertain" now covers a much wider range of visually-similar-but-different
materials (e.g. white ceramic vs Glass, cardboard vs Paper-adjacent Cloth
tags). Re-run your debug_crops/ + probability-breakdown workflow specifically
against borderline items across the new classes before trusting live demo
behavior -- same approach that caught the transparent-plastic issue before.
