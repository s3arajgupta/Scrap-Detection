# ♻️ Physical Scrap Verification Engine (Temporal CV Dashboard)

An autonomous, zero-model-training computer vision verification pipeline and Streamlit dashboard that connects physical scrap identification with marketplace intake workflows.

The system ingests 10-second inspection video clips, samples frames at 1-second intervals (1 FPS, 10 samples total), classifies each frame using pre-trained zero-shot vision intelligence, and calculates a commercial intake decision via **Majority Voting with Deep Explainability & Prediction Transparency**.

---

## 🔍 Explainability & Decision Transparency Features

The dashboard bridges raw computer vision outputs with auditable, commercial marketplace policies:

### 1. Commercial Decision Transparency & Policy Engine
- **Risk Rating:** Categorizes incoming lots into `LOW RISK`, `MODERATE RISK`, or `HIGH AUDIT RISK`.
- **Marketplace Action Recommendations:** Generates automated commercial actions (e.g., *"Auto-Approve Lot: Release for instant marketplace trading at standard index pricing"* or *"Hold for Conditional Verification: Apply 5-10% deduction"*).
- **Automated Policy Breakdown:** Formulates explicit decision rationales checking:
  - **Temporal Coherence:** Percentage of frames uniformly matching the dominant scrap class ($\ge 80\%$).
  - **Continuous Stability Run:** Longest uninterrupted sequence of matching frames (requires $\ge 4\text{s}$ continuous run to reject transient camera jitter).
  - **Transient Noise Filtering:** Validates that short drops (< 3s) are transient handling occlusions rather than bulk material contamination.

### 2. Frame-by-Frame Prediction Transparency
- **Confidence Separation Margin ($\Delta$):** Each 1-second sample box displays its confidence delta over the runner-up class (e.g., `Copper 92% (Δ +86%)` over `Brass 6%`), proving prediction certainty.
- **Interactive Visual Evidence Inspector:** Select any sample second ($0\text{s} - 9\text{s}$) to inspect the exact physical cues driving the prediction:
  - Chrominance and color space analysis (e.g., red-orange copper reflectance vs. yellow-gold brass).
  - Surface luster and specular reflections.
  - Presence or absence of non-metallic occlusions or oxide coatings.

### 3. Full Temporal Audit Trail
- Includes an auditable tabular log of all 10 inspection seconds logging timestamps, primary predictions, confidence scores, dominant agreement status, runner-up classes, and separation margins.

---

## 📁 Datasets Directory (`.\datasets`)

Place your local test videos, scrap clips, and dataset footage inside the [datasets/](file:///e:/GitHub/Maleo/datasets/) folder:

```text
Maleo/
├── datasets/                   # 📂 Place your inspection clips and datasets here
│   ├── README.md               # Folder guidelines and format specs
│   ├── copper/                 # 10s clips of copper wire, tubing, coils
│   ├── aluminum/               # 10s clips of aluminum extrusions, sheet, turnings
│   ├── brass/                  # 10s clips of brass scrap
│   ├── steel/                  # 10s clips of HMS steel, structural rebar
│   └── mixed_debris/           # 10s clips of mixed, occluded, or contaminated lots
```

### Supported Video Formats:
- **Formats:** `.mp4`, `.mov`, `.avi`
- **Clip Duration:** 10 seconds (1 frame extracted per second).
- **Upload at Runtime:** Use the drag-and-drop uploader on the dashboard to test clips directly from `.\datasets\`.

---

## 🛠️ Installation Guide

### 1. Open Terminal in the Project Directory
```powershell
cd e:\GitHub\Maleo
```

### 2. Create a Virtual Environment
```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If script execution is blocked on Windows, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

- **Windows (Command Prompt):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit dashboard:

```powershell
streamlit run app.py
```

Or run directly using the virtual environment executable:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Once launched, open your web browser at:
👉 **`http://localhost:8501`**

---

## 📁 Project Structure

```text
Maleo/
├── .venv/                      # Python virtual environment
├── datasets/                   # 📂 Sample scrap clips & inspection video datasets
│   └── README.md
├── requirements.txt            # Streamlit, Transformers, Ultralytics, OpenCV, Altair, etc.
├── app.py                      # Streamlit UI (Majority voting, 10-box grid, Inspector, Audit Log)
├── verifier_engine.py          # Zero-shot classification, decision policy, & explainability cues
├── video_processor.py          # OpenCV 1 FPS temporal frame sampler
├── IMPLEMENTATION_DOSSIER.md   # Complete system specification and technical dossier
└── README.md                   # Installation, transparency architecture, and dataset guide
```
