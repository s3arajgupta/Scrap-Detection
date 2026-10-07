# Technical Dossier: Temporal Scrap Verification Dashboard

## 1. System Overview & Core Objectives

### 1.1 Objective

Build an autonomous, zero-model-training verification pipeline and Streamlit dashboard that ingests physical scrap video clips (10-second inspection clips), extracts frames at 1-second intervals (1 FPS, 10 samples total), classifies each frame using pre-trained zero-shot vision intelligence, and synthesizes an intake decision via **Majority Voting with Policy Explainability & Prediction Transparency**.

### 1.2 Pipeline Architecture

```
                       [ Input Scrap Video Clip (10s) ]
                                      │
                         [ OpenCV 1 FPS Sampler ]
                     (Extracts 10 discrete frame samples)
                                      │
                   [ Pre-trained Zero-Shot Classifier ]
                      (CLIP ViT / Calibrated Engine)
               Scores each frame & extracts visual feature cues:
            • Copper scrap            • Aluminum scrap
            • Steel/Iron scrap        • Brass scrap
            • Electronic waste        • Plastic/Debris contamination
                                      │
                                      ▼
                        [ Majority Voting Engine ]
               • Identifies Dominant Material across 10 samples
               • Calculates Persistence Ratio (e.g. 8/10s = 80%)
               • Computes Longest Consecutive Stability Run
               • Evaluates Risk Rating & Marketplace Action
               • Synthesizes Policy Rationale & Runner-up Margins
                                      │
                                      ▼
             [ Streamlit Temporal Verification UI Dashboard ]
               • Row 1: Video Player & Decision Transparency Card
               • Row 2: 10 Dedicated Sample Boxes (Thumbnails + Margin Δ)
               • Row 3: Interactive Frame-by-Frame Explainability Inspector
               • Row 4: Trajectory Chart, Dissonance Report & Audit Log
```

---

## 2. Updated UI Wireframe & Component Breakdown

```
+------------------------------------+-----------------------------------------------------+
|                                    | DECISION & TRANSPARENCY POLICY                      |
|                                    | [ STRONG MATCH ] [Rating: LOW RISK] (80% persistence)
|                                    | Dominant Material: Copper scrap                     |
|        [ VIDEO PLAYER ]            | Agreement: 8/10s | Max Run: 5s continuous | Mean: 0.89
|        input_clip.mp4 (10 s)       | ⚡ Marketplace Action: Auto-Approve Lot for trading  |
|                                    | [🛡️ Decision Rationale & Policy Breakdown]          |
|                                    | [▼ Top-5 Probability Distribution]                  |
+------------------------------------+-----------------------------------------------------+
| PER-SECOND EVIDENCE (10-Second Temporal Localization — 1 Box per Sample)                 |
|   0s          1s          2s          3s          4s          5s     ...     9s          |
| [THUMB]     [THUMB]     [THUMB]     [THUMB]     [THUMB]     [THUMB]        [THUMB]       |
| Copper      Copper      Copper      Debris      Debris      Copper         Copper        |
|   92%         94%         89%         65%         68%         91%            90%         |
|  Δ +86%      Δ +88%      Δ +81%      Δ +43%      Δ +46%      Δ +84%         Δ +85%       |
+------------------------------------------------------------------------------------------+
| 🔬 FRAME-BY-FRAME EXPLAINABILITY & VISUAL EVIDENCE INSPECTOR (Expandable Panel)          |
| * Selectable frame slider (0s - 9s)                                                      |
| * Primary Prediction vs. Runner-Up Hypothesis with confidence margin delta               |
| * Specific visual evidence drivers (chrominance, luster, texture striations, etc.)      |
+------------------------------------------------------------------------------------------+
| EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS (Expandable Panel)                           |
| * Second-by-Second Confidence Trajectory Line Chart (Dominant Material vs 0.75 Baseline) |
| * Semantic Dissonance Diagnostic (Edge Case & Transient Drop Explanations)               |
| * 📊 Temporal Audit Trail & Frame-by-Frame Transparency Log (Data Table)                 |
+------------------------------------------------------------------------------------------+
```

---

## 3. Decision Transparency & Policy Rules

The decision engine applies an automated policy to translate temporal predictions into commercial trading outcomes:

| Metric | Rule Threshold | Policy Meaning |
| :--- | :--- | :--- |
| **Temporal Persistence** | $\ge 80\%$ | High lot purity; uniform scrap lot. |
| **Longest Consecutive Run** | $\ge 4\text{s}$ uninterrupted | Guarantees temporal stability; filters out jitter. |
| **Confidence Margin ($\Delta$)** | $> +40\%$ over runner-up | High classification certainty; distinct alloy signature. |
| **Contamination Exposure** | $\le 20\%$ | Transient noise allowed without disqualifying the lot. |

### Commercial Actions Output

- **`STRONG MATCH` (Low Risk):** Auto-approves the lot for instant B2B marketplace listing at standard index pricing.
- **`MODERATE MATCH` (Moderate Risk):** Holds for conditional verification, flags a 5-10% contamination deduction, or requests weighbridge scale slips.
- **`INCONCLUSIVE` (High Risk):** Rejects automated intake and routes to a certified yard inspector for manual physical sorting.

---

## 4. Data Schemas (Pydantic Models)

```python
from pydantic import BaseModel
from typing import List, Dict

class PerSecondSample(BaseModel):
    timestamp_sec: int
    predicted_class: str
    confidence: float
    is_dominant_match: bool
    runner_up_class: str
    runner_up_confidence: float
    confidence_margin: float
    visual_evidence_cues: List[str]
    top_probabilities: Dict[str, float]

class MajorityVoteDecision(BaseModel):
    dominant_material: str
    persistence_ratio: str
    persistence_percentage: float
    longest_consecutive_run: int
    contamination_rate: float
    avg_confidence: float
    decision_badge: str
    risk_rating: str
    marketplace_action: str
    decision_rationale: List[str]
    top_5_aggregate: Dict[str, float]

class DiagnosticReport(BaseModel):
    trajectory: List[float]
    dissonance_explanation: str
    transparency_summary: str

class VerificationResult(BaseModel):
    video_filename: str
    decision: MajorityVoteDecision
    samples: List[PerSecondSample]
    diagnostics: DiagnosticReport
```

---

## 5. Execution Instructions

Launch the dashboard from your virtual environment:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
