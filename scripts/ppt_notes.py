"""Speaker script for the 15-minute report, written into the slide notes.

    python scripts/ppt_notes.py [deck.pptx]

Run this **last** -- after ppt_tail_redesign.py -- because the tail builder
rebuilds 18-23 from v3 and would drop the notes. The same text is also dumped
to results/ppt/speaker_notes.txt so it can be read (or printed) without
opening PowerPoint.

Budget: the '[~xx s]' tags sum to ~14 min, leaving a minute for transitions.
Every number here is one that survives checking against results/summary.md.
"""
from __future__ import annotations

import pathlib
import shutil
import sys

from pptx import Presentation

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_TXT = ROOT / "results" / "ppt" / "speaker_notes.txt"
DECK = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    r"E:\大作业\GPU-Parallel Batched Kalman Smoothing and LoRA Fine-Tuned "
    r"Time-Series Foundation Model-v4.pptx")

NOTES = {
    1: """[~30 s]
Good morning. My name is Guo Haisu. The question in this fifteen-minute report is simple: when the GNSS fix disappears for a minute, how much of the lost accuracy can we win back — and what does it cost at big-data scale?

Three parts: a physics-based smoother, a GPU pipeline that makes it fast, and a learned module with a gate.""",

    2: """[~20 s]
Four chapters: the problem and the data; the filter and the GPU batching, which is the engineering core; the time-series model and LoRA; and the map prior, where the three headline numbers come together.""",

    3: """[~5 s]
Chapter one — why this problem exists.""",

    4: """[~55 s]
Three tiers, each removing what the one before cannot. Tier one is physics: an error-state Kalman filter with RTS smoothing, which cuts the outage drift by 90.5 percent — 104.65 metres to 9.93. Tier two is systems: the same smoother batched across windows on CUDA, so the full 8.64-million-row corpus finishes in seventeen minutes on one RTX 4090, 48.6 times the serial CPU time on a million rows. Tier three is learning plus the map: a LoRA-tuned model behind a safety gate, and a road-centreline prior.

Every tier is judged only on routes it never saw.""",

    5: """[~45 s]
Here is the data. Eight point six four million rows — 48 routes, half an hour each at 100 hertz. Two properties matter. The rate gap: IMU at 100 hertz against GNSS at 1 hertz, a hundred to one. And the failure window: a 60-second outage is six thousand uncorrected inertial steps.

The corpus comes from our own analytic simulator with a documented noise model, so everything here is reproducible; replaying a real IMU log is the first thing we would do next.""",

    6: """[~50 s]
Why is one minute so destructive? Mechanisation integrates twice. An accelerometer bias of 0.05 metres per second squared, integrated twice over 60 seconds, is already ninety metres. And it gets worse: a gyro bias tilts the attitude matrix, gravity leaks into the horizontal channel, and the error grows cubically.

That is what the filter has to fight — and why pure dead reckoning reaches hundreds of metres before the fix comes back.""",

    7: """[~5 s]
Chapter two — the filter, and making it fast.""",

    8: """[~60 s]
The filter carries fifteen error states. Three steps. At 100 hertz we propagate the nominal state and the covariance. At 1 hertz a GNSS fix updates the hidden states — attitude and IMU biases — through the off-diagonal covariance blocks. Then the closed-loop reset: the error estimate is injected and zeroed.

Two choices matter for stability. We estimate a three-parameter attitude error instead of the quaternion, so the covariance never goes singular, and we use the Joseph form, which keeps it positive definite over millions of steps. Forward filtering alone halves the drift: 104.65 to 51.44.""",

    9: """[~55 s]
Smoothing is where the gain is. A filter only uses the past; once the outage ends, the returning fix reveals how far you drifted, and an RTS pass pushes that knowledge backwards in time.

Two details. We cache posterior and predicted pairs and solve for the gain instead of inverting matrices. And there is a trap: because the ESKF resets its error state at every step, smoothing the error state returns exactly zero — so we smooth the full state. That is what takes 51.44 to 9.93, ninety percent off dead reckoning.""",

    10: """[~45 s]
Why is this an HPC problem? Each step is a chain of small 15-by-15 products — microseconds of arithmetic with almost no data reuse, so serial execution wastes the device.

The fix is not making one step faster; it is putting thousands of independent steps in flight. So the whole design question becomes: where is the independent axis?""",

    11: """[~50 s]
The independent axis is the window. Routes are cut into overlapping windows, windows run in parallel, and the recursion stays serial only inside a window.

Three rules make it safe. The burn-in — ninety seconds — must exceed the outage, so every output sample has valid fixes on both sides. A two-pass warm start begins each window from the previous one's state. And tent-shaped overlap-add makes the joins seamless to about one percent of a step. The price of parallelism is eight point two percent: 10.75 metres instead of 9.93.""",

    12: """[~50 s]
Two experiments. The burn-in ablation: with a 30-second burn-in, route 43's outage begins just before the output boundary, the window ends inside the gap, and the error jumps to 133 metres; ninety seconds restores parity. W is a requirement, not a knob.

The benchmark: a million rows take 2,245 seconds on the CPU and 46.2 on the GPU — 48.6 times — and 177.6 times once the chunk grows to 500. The gain grows with batch size because the kernel is launch-bound when batches are small.""",

    13: """[~5 s]
Chapter three — the learned module.""",

    14: """[~50 s]
Smoothing leaves about ten metres, and that residual is structured — multipath, scale-factor error, thermal drift. We make it supervised: twenty channels at ten hertz, a fifty-one-second context, predicting the residual in the middle.

The channel design is the argument. Raw IMU carries the dynamics. Log-compressed innovation stops the reacquisition spike from dominating the normaliser. Outage timing encodes how long the bias has been integrating. And the filter states — velocity and biases — are the identifiability key: without them, a long window to a ten-metre residual is under-determined.""",

    15: """[~40 s]
Two models. B4, a two-layer bi-LSTM trained from scratch, is the control. B6 is GPT4TS: a pretrained distilgpt2 backbone, 82.8 million parameters, with the sensor windows patched into tokens.

The reason to use attention is the span — it links the reacquisition spike at the end of an outage to the six hundred inertial steps at its start, without the vanishing gradient a recurrent model pays over that distance.""",

    16: """[~40 s]
Full fine-tuning is the wrong tool here. With 82.8 million parameters and a noisy residual, it memorises the training routes in a few epochs. LoRA freezes the backbone and learns a rank-8 update — A times B — inside the attention projections.

We train 1.06 percent of the parameters, the rank-8 bottleneck regularises, and at inference the update merges back into the weights, so there is no latency cost.""",

    17: """[~55 s]
One design decision, one safety mechanism. The decision: windows weighted twice, outage steps eleven times, so 82 percent of the gradient lands on the 7 percent of the timeline where the filter is blind.

The mechanism is the gate. Both models train cleanly — 5.30 to 4.40 for the LSTM, 4.98 to 1.94 for GPT4TS. On held-out routes the best validation error is 11.79 and 11.68, against 11.48 for predicting zero. Unguarded, the scale factor buys two to three percent on validation and loses two to five on test — so a one-standard-error rule returns alpha equal to zero on all eight routes. The module only acts where it can prove a gain.""",

    18: """[~55 s]
So what did it buy? On the left, the honest scoreboard: dead reckoning 104.65, filter 51.44, smoothing 9.93, GPU 10.75 — a ninety percent reduction, and the worst single outage from 413 metres to 28.

On the right, where the rest of the error lives. Eighty-four percent of it is a smooth quadratic in outage time — a frozen initial-state error, not noise. Ninety-one percent of it disappears if that smooth part is known exactly — that is, if the missing fix existed. An oracle IMU calibration adds 0.9 percent. The trained corrector adds none of the remainder, because direction transfers across routes but amplitude does not: the offset is fixed at outage entry, where no fix is recorded. That is an information boundary — and the gate is how we report it.""",

    19: """[~5 s]
Chapter four — what spatial information adds.""",

    20: """[~50 s]
Physics and learning both work in time. The map works in space. A centreline supplies exactly the component the filter cannot observe — cross-track error.

We project the outage onto it with continuity constraints: arc length must increase, speed must stay under the limit, heading must agree. That takes 9.93 to 6.30 — 36.5 percent — on all eight test routes. Nearest-point projection is not enough: it reaches 7.52 and regresses the self-intersecting route 46 by 36 percent. And it tolerates about a metre of survey error; at two metres, route 46 jumps to a parallel branch.""",

    21: """[~45 s]
And downstream: a lane line is observed relative to the vehicle, so a pose error becomes a map error — the doubled, smeared lane lines of a map built from uncorrected trajectories.

Lateral offset: 43.9 metres from dead reckoning, 6.5 from the filter, 0.28 with the prior, and 86 percent of points inside the half-metre HD tolerance. One caveat, honestly: 0.28 is the chord error of a one-hertz centreline, not a pose accuracy. With a prior map, this measures map update — not mapping from scratch.""",

    22: """[~50 s]
Three numbers, three different limits. 48.6 times is compute — how fast one GPU filters a million rows. 90.5 percent is geometry and information — how much the smoother can recover, and the eight percent we pay for parallelism. 0.28 metres is mapping — what the prior does to the last observable component.

What comes next: replay a real IMU log, because this corpus is simulated; widen the window, since ninety seconds is a requirement, not the window length; and feed the corrector what the outage removed — until then the gate keeps alpha at zero.""",

    23: """[~15 s]
That is the report. Thank you — I am happy to take questions.""",
}


def main() -> None:
    prs = Presentation(DECK)
    assert len(NOTES) == len(prs.slides._sldIdLst), \
        f"{len(NOTES)} notes for {len(prs.slides._sldIdLst)} slides"
    for i, note in NOTES.items():
        prs.slides[i - 1].notes_slide.notes_text_frame.text = note

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(
        "\n\n".join(f"--- slide {i} ---\n{n}" for i, n in NOTES.items()),
        encoding="utf-8")

    backup = DECK.with_name(DECK.stem + ".pre-notes.pptx")
    shutil.copy2(DECK, backup)
    prs.save(DECK)
    print(f"notes written to {len(NOTES)} slides")
    print(f"plain text -> {OUT_TXT}")
    print(f"backup     -> {backup.name}")


if __name__ == "__main__":
    main()
