# Technical Dossier: Temporal Scrap Verification Dashboard

## 1. System Overview & Core Objectives

### 1.1 Objective
Build a lightweight, zero-model-training verification pipeline and Streamlit dashboard that ingests scrap video clips (e.g., 10 seconds), samples frames at 1-second intervals (1 FPS), evaluates material correspondence using pre-trained zero-shot vision intelligence, and presents:
1. **Aggregate Decision & Top-5 Probabilities**
2. **10-Second Temporal Localization Matrix** (`[██]` / `[  ]` per-second presence)
3. **Explainability & Counterfactual Diagnostics** (modality attribution, cosine similarity trajectory, semantic dissonance root cause)

```
                       [ Input Video Clip (10s) ]
                                   │
                       [ OpenCV 1 FPS Sampler ]
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
     [ Zero-Shot Embedder ]                 [ VLM Reasoning Engine ]
   SigLIP / OpenCLIP ViT-B-16                 Gemini 2.5 Flash
   • Second-by-second cosine similarity      • Contamination % & Dissonance
   • Top-5 category distribution             • Modality attribution
                └──────────────────┬──────────────────┘
                                   │
                        [ Aggregated JSON Schema ]
                                   │
                                   ▼
             [ Streamlit Temporal Verification UI Dashboard ]
```

---

## 2. Component Breakdown & Wireframe Mapping

```
+------------------------------------+-----------------------------------------------------+
|                                    | DECISION                                            |
|                                    | [ STRONG MATCH ] (0.91 correspondence)              |
|                                    | Material: Copper (0.88) [▼ Top-5 Probability Drawer]|
|        [ VIDEO PLAYER ]            +-----------------------------------------------------+
|        input_clip.mp4 (10 s)       |                                                     |
+------------------------------------+-----------------------------------------------------+
| PER-SECOND EVIDENCE (10-Second Temporal Localization)                                    |
| Timestamps:      0s     1s     2s     3s     4s     5s     6s     7s     8s     9s       |
| Visual: Copper  [██]   [██]   [██]   [  ]   [  ]   [██]   [██]   [██]   [██]   [██]      |
+------------------------------------------------------------------------------------------+
| EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS (Expandable Panel)                           |
| * Modality Attribution: Visual Branch 86%                                                |
| * Second-by-Second Cosine Similarity Trajectory                                          |
| * Semantic Dissonance Diagnostic (Edge Case Explanations)                                |
+------------------------------------------------------------------------------------------+
```

| UI Section | Visual Component | Underlying Logic / Data Source |
| :--- | :--- | :--- |
| **Top Left** | Video Player (`st.video`) | Displays uploaded or sample video with native controls. |
| **Top Right** | Decision Badge & Material Grade | Threshold check on aggregate score ($\ge 0.85 \rightarrow$ `STRONG MATCH`, $0.70-0.84 \rightarrow$ `MODERATE MATCH`, $< 0.70 \rightarrow$ `MISMATCH`). |
| **Top Right** | Top-5 Probability Drawer (`st.expander`) | Softmax distribution across candidate scrap classes. |
| **Middle** | 10-Second Temporal Localization Grid | 10 column blocks ($0\text{s} - 9\text{s}$). Green/filled `[██]` if per-second score $\ge \tau$, blank `[  ]` if occluded or mismatched. |
| **Bottom** | Modality Attribution Metric | Visual contribution ratio vs. Metadata/Prior claim confidence. |
| **Bottom** | Cosine Similarity Trajectory Chart | Altair line chart tracking confidence fluctuation across $0\text{s} - 9\text{s}$ with decision threshold line. |
| **Bottom** | Semantic Dissonance Diagnostic | Root cause explanation for anomalous drops in confidence (e.g., occlusion, motion blur, foreign matter). |

---

## 3. Data Schema (Pydantic Models)

- `PerSecondEvidence`: Timestamp, confidence score, binary match state, detected category, dissonance notes.
- `DiagnosticReport`: Visual branch weight, metadata branch weight, trajectory scores, semantic dissonance notes.
- `ScrapVerificationResult`: Overall decision badge, correspondence score, primary material, Top-5 probabilities, timeline list, diagnostic report.

---

## 4. Zero-Training Architecture Options

1. **Option A: Pre-trained VLM (Gemini 2.5 Flash / GPT-4o-mini)**
   - Ingests all 10 sampled frames in one batch prompt.
   - Zero training required.
   - Extracts scrap grades (ISRI standard), contamination %, and root cause dissonance in structured JSON.

2. **Option B: Local Zero-Shot (SigLIP / OpenCLIP `ViT-B-16`)**
   - Encodes domain prompt texts (e.g. `"pure bright bare copper wire"`, `"insulated wire with PVC"`, `"aluminum 6063 extrusions"`).
   - Computes cosine similarity per frame in $< 50\text{ms}$ locally on CPU/GPU.
