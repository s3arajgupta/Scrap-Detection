# Technical Dossier: Temporal Scrap Verification Dashboard

## 1. System Overview & Core Objectives

### 1.1 Objective
Build an autonomous, zero-model-training verification pipeline and Streamlit dashboard that ingests physical scrap video clips (10-second inspection clips), extracts frames at 1-second intervals (1 FPS, 10 samples total), classifies each frame using pre-trained zero-shot vision intelligence, and synthesizes an aggregate intake decision via **Majority Voting**.

### 1.2 Pipeline Architecture

```
                       [ Input Scrap Video Clip (10s) ]
                                      │
                         [ OpenCV 1 FPS Sampler ]
                     (Extracts 10 discrete frame samples)
                                      │
                   [ Pre-trained Zero-Shot Classifier ]
                      (CLIP ViT / Calibrated Engine)
               Scores each frame against canonical scrap classes:
            • Copper scrap            • Aluminum scrap
            • Steel/Iron scrap        • Brass scrap
            • Electronic waste        • Plastic/Debris contamination
                                      │
                                      ▼
                        [ Majority Voting Engine ]
               • Identifies Dominant Material across 10 samples
               • Calculates Persistence Ratio (e.g. 8/10s = 80%)
               • Computes Mean Confidence & Decision Badge
               • Aggregates Top-5 Probability Distribution
                                      │
                                      ▼
             [ Streamlit Temporal Verification UI Dashboard ]
               • Row 1: Video Player & Majority Voting Decision
               • Row 2: 10 Dedicated Sample Boxes (with Frame Thumbnails)
               • Row 3: Trajectory Chart & Semantic Dissonance Diagnostic
```

---

## 2. Updated UI Wireframe & Component Breakdown

```
+------------------------------------+-----------------------------------------------------+
|                                    | DECISION (Majority Voting)                          |
|                                    | [ STRONG MATCH ] (80% persistence)                  |
|                                    | Dominant Material: Copper scrap                     |
|        [ VIDEO PLAYER ]            | Temporal Agreement: 8 / 10 seconds | Mean: 0.89     |
|        input_clip.mp4 (10 s)       | [▼ Top-5 Probability Drawer]                        |
|                                    +-----------------------------------------------------+
+------------------------------------------------------------------------------------------+
| PER-SECOND EVIDENCE (10-Second Temporal Localization — 1 Box per Sample)                 |
|   0s          1s          2s          3s          4s          5s     ...     9s          |
| [THUMB]     [THUMB]     [THUMB]     [THUMB]     [THUMB]     [THUMB]        [THUMB]       |
| Copper      Copper      Copper      Debris      Debris      Copper         Copper        |
|   92%         94%         89%         65%         68%         91%            90%         |
+------------------------------------------------------------------------------------------+
| EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS (Expandable Panel)                           |
| * Second-by-Second Confidence Trajectory Line Chart (Dominant Material vs 0.75 Baseline) |
| * Semantic Dissonance Diagnostic (Edge Case & Transient Drop Explanations)               |
+------------------------------------------------------------------------------------------+
```

### Component Details:
| Section | Component | Functionality |
| :--- | :--- | :--- |
| **Top** | Main Video Uploader (`st.file_uploader`) | Accepts `.mp4`, `.mov`, `.avi` inspection clips directly on the main page. |
| **Row 1 (Left)** | Video Player (`st.video`) | Displays uploaded clip with native playback and scrubber. |
| **Row 1 (Right)** | Majority Voting Decision Panel | Displays Dominant Material, Decision Badge (`STRONG MATCH` $\ge 70\%$, `MODERATE MATCH` $\ge 50\%$, `INCONCLUSIVE`), Persistence Ratio, Mean Confidence, and collapsible Top-5 distribution drawer. |
| **Row 2** | 10 Sample Evidence Boxes | 10 individual cards (seconds $0\text{s} - 9\text{s}$). Each card renders: (1) timestamp header, (2) extracted frame thumbnail, (3) predicted scrap class, (4) confidence score, and (5) green indicator for dominant agreement or amber for divergent/contaminated frames. |
| **Row 3** | Confidence Trajectory Chart | Altair line chart plotting second-by-second confidence against a 0.75 decision threshold baseline (`width="stretch"`). |
| **Row 3** | Semantic Dissonance Diagnostic | Natural language explanation identifying anomalous frames (e.g., operator glove occlusion, specular reflection, or foreign debris). |

---

## 3. Data Schema & State Contracts (Pydantic Models)

```python
from pydantic import BaseModel, Field
from typing import List, Dict

class PerSecondSample(BaseModel):
    timestamp_sec: int
    predicted_class: str
    confidence: float
    is_dominant_match: bool
    top_probabilities: Dict[str, float]

class MajorityVoteDecision(BaseModel):
    dominant_material: str
    persistence_ratio: str            # e.g., "8 / 10 seconds"
    persistence_percentage: float     # e.g., 80.0
    avg_confidence: float
    decision_badge: str               # "STRONG MATCH", "MODERATE MATCH", "INCONCLUSIVE"
    top_5_aggregate: Dict[str, float]

class DiagnosticReport(BaseModel):
    trajectory: List[float]
    dissonance_explanation: str

class VerificationResult(BaseModel):
    video_filename: str
    decision: MajorityVoteDecision
    samples: List[PerSecondSample]
    diagnostics: DiagnosticReport
```

---

## 4. Zero-Training Classification Engine

To avoid model training and data annotation, the classification pipeline uses a **two-tier zero-shot approach**:

1. **Tier 1: Pre-Trained Zero-Shot Vision Transformer (Hugging Face / CLIP)**
   - Model: `openai/clip-vit-base-patch32` or `google/siglip-base-patch16-224`.
   - Ingests each frame and computes text-image cosine similarities against candidate scrap classes without fine-tuning.

2. **Tier 2: Calibrated Visual Feature Fallback**
   - High-speed visual colorimetry and texture heuristics (calibrated to Copper [red-orange], Brass [yellow-gold], Aluminum [bright silver], Steel [dark gray/oxide], E-waste [green/multi-color], Debris [mixed low-saturation]).
   - Guarantees immediate zero-latency evaluation even when offline or during initial model loading.

---

## 5. File Structure & Responsibilities

```
e:\GitHub\Maleo\
├── .venv/                      # Isolated Python 3.14 virtual environment
├── requirements.txt            # Streamlit, Transformers, Ultralytics, OpenCV, Altair, etc.
├── app.py                      # Main Streamlit dashboard (Majority voting UI & 10-box grid)
├── verifier_engine.py          # Zero-shot classification, majority voting, & diagnostics
├── video_processor.py          # OpenCV 1 FPS frame extraction utility
└── IMPLEMENTATION_DOSSIER.md   # This architecture and design document
```

---

## 6. Execution Instructions

Launch the dashboard from your virtual environment:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
