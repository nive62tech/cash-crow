# Smart Bin — Phase 1: Detection, Hand-Held Object Isolation & ROI Selection

Smart Bin is an AI-powered waste sorting system: a camera detects a waste
item and the correct compartment (Plastic / Paper / Other) opens
automatically. Final deployment target is a Raspberry Pi 4/5 with a
Pi Camera Module 3.

## Demo

<!--
  Record a short screen capture of detect.py running (a person appearing/
  disappearing, then holding an object so the red box appears and holds
  steady) and drop the video file here, e.g.:

  https://github.com/<your-username>/<your-repo>/assets/<asset-id>/demo.mp4

  Easiest way to get that link: open a GitHub Issue or PR on your repo,
  drag-and-drop the video file into the comment box, wait for it to
  upload, then copy the generated URL it inserts -- paste that URL below
  instead of writing the video into the repo directly (keeps repo size
  small). GitHub will render it as a playable video inline.
-->
*(Demo video goes here.)*

## What Phase 1 proves

This phase does **not** classify waste into Plastic/Paper/Other yet — there's
no custom dataset or trained model for that (that's Phase 3). Instead, Phase 1
proves the full detection + decision pipeline works end-to-end on a laptop
webcam, using YOLOv8n's **pre-trained COCO weights**:

1. Real-time object detection works on this hardware/software stack.
2. A second model (MediaPipe Hand Landmarker) finds hand position each frame,
   independent of YOLO — since COCO has no "hand" class at all.
3. Combining both: the system can correctly tell "an object is being held"
   apart from a person just standing in frame, or an empty scene.
4. A bounding-box distance heuristic ranks multiple detections by how close
   they likely are to the camera, with no depth sensor.
5. A display/state layer shows the right indicator (person / object /
   nothing) with sensible timing — not flickering on every noisy frame.

## Why this design: the three-part pipeline

- **YOLOv8n** — object detection: draws a box around anything it recognizes.
  Chosen for being lightweight enough to eventually run on a Raspberry Pi.
- **MediaPipe Hand Landmarker** — hand position tracking, a completely
  separate model, because YOLO/COCO has no concept of a "hand" at all.
- **Custom logic (this repo's code)** — decides what actually matters:
  checks whether a detected object's box overlaps a hand's region (→ "held"),
  scores multiple detections by closeness, and drives a small state machine
  for what to display and for how long.

## Display rules

| Situation | What shows |
|---|---|
| Something is held by fingers/palm | **Red** box, labeled "Object" — persists continuously while held |
| No held object, a person is visible | **Light green** box on the person, only for the first 2 seconds after they appear |
| Nothing detected at all | **Blue** "scanning" zone, for the first 5 seconds of empty scene |

A person or empty-scene box only reappears after a **genuine absence** —
4 seconds for a person, so ordinary movement within frame isn't mistaken for
"a new person arrived." A held object's box is designed to persist through
brief single-frame detection flicker rather than blinking on/off.

## Real obstacles hit during Phase 1 (and what they taught us)

**1. COCO's closed vocabulary.** YOLOv8n-COCO can only output one of its 80
trained classes — ever — even for objects it's never seen (a candy wrapper
got called "cell phone", a notepad got called "book"). This isn't a bug or
something a config change fixes; it's the nature of a closed-set classifier,
which has no "unknown" output. This is exactly why the on-screen label for
held objects is intentionally kept generic ("Object") rather than pretending
to guess a real material type — that's reserved for Phase 3's custom-trained
classifier.

**2. A breaking change in MediaPipe itself.** Mid-build, the documented
`mp.solutions.hands` API stopped working, because Google deprecated it in a
recent release in favor of a newer "Tasks" API. Confirmed via other
developers hitting the identical error around the same time — not a mistake
in this code, a genuine library-side change that needed a rewrite of the
hand-detection call.

**3. Detection confidence flicker breaking display timers.** YOLO's
confidence for the same, unmoving object jitters frame to frame. An early
version treated every single missed frame as "it's gone," which kept
resetting timers to zero — so a box that was supposed to disappear after a
few seconds appeared to "never" disappear. Fixed with a grace period: a
category must be truly absent for a full window before being treated as
gone, not reactive to one noisy frame.

**4. Overly generous hand-region padding causing false positives.** To
catch large objects only partially gripped by fingers, the "held" zone
around each hand was padded generously — too generously, as it turned out:
it started catching things merely *near* the hand (like the user's own
face) as "held." Tightened to require closer, more genuine finger/palm
proximity.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python detect.py
```

Press `q` to quit. The terminal prints per-frame debug info (raw YOLO
guesses, hand count, and which box was drawn) useful for troubleshooting;
none of this is shown on the video window itself.

## What's NOT in Phase 1 (by design)

- No custom Plastic/Paper/Other classification — needs a labeled dataset,
  which is what `data/` is reserved for in Phase 2/3
- No Raspberry Pi / Pi Camera code yet — laptop webcam only
- No compartment/servo control logic yet

## Repo structure

```
smart-bin/
├── data/                    # empty for now -- will hold labeled Plastic/Paper/Other images later
├── src/
│   ├── detect.py            # webcam capture, YOLO + hand detection, display state machine
│   └── roi_selector.py      # bounding-box distance heuristic
├── requirements.txt
└── README.md
```