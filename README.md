# ♻️ Physical Scrap Verification Engine (Temporal CV Dashboard)

An autonomous, zero-model-training computer vision verification pipeline and Streamlit dashboard that connects physical scrap identification with marketplace intake workflows.

The system ingests 10-second video inspection clips, samples frames at 1-second intervals (1 FPS, 10 samples total), classifies each frame using pre-trained zero-shot vision intelligence, and calculates an intake decision via **Majority Voting**.

---

## 🚀 Features

- **1 FPS Temporal Frame Extraction:** Samples exactly 10 frames across a 10-second inspection clip using OpenCV.
- **Pretrained Zero-Shot Classification:** Categorizes scrap materials without manual model training into:
  - `Copper scrap`
  - `Aluminum scrap`
  - `Steel/Iron scrap`
  - `Brass scrap`
  - `Electronic waste (E-waste)`
  - `Plastic / Debris contamination`
- **1 Box per Sample (10 Sample Boxes):** Displays a dedicated card for each 1-second frame with its image thumbnail, predicted scrap label, confidence score, and status indicator.
- **Majority Voting Decision Engine:** Aggregates temporal predictions to determine the **Dominant Material**, persistence ratio (e.g., `8 / 10 seconds = 80%`), and mean confidence.
- **Explainability & Diagnostics:** Features an Altair second-by-second confidence trajectory chart and semantic dissonance diagnostics for anomalous frames (occlusions, reflections, or contamination).

---

## 📋 Prerequisites

- **Python 3.10+** (Tested on Python 3.10 – 3.14)
- **Windows PowerShell, Command Prompt, or Bash**

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
├── requirements.txt            # Project dependencies (Streamlit, Transformers, Ultralytics, OpenCV, Altair)
├── app.py                      # Main Streamlit UI dashboard
├── verifier_engine.py          # Zero-shot classification, majority voting, & diagnostics
├── video_processor.py          # OpenCV 1 FPS temporal frame sampler
├── IMPLEMENTATION_DOSSIER.md   # Complete system specification and technical dossier
└── README.md                   # Installation and usage instructions
```

---

## 💡 How to Use

1. **Upload Video:** Click the upload box at the top to select a 10-second scrap inspection clip (`.mp4`, `.mov`, `.avi`).
2. **Preview:** View the clip playing in the left panel.
3. **Review Majority Vote:** Check the right panel for the overall **Dominant Material**, persistence %, and top-5 probability distribution.
4. **Inspect Per-Second Boxes:** Look through the 10 sample boxes ($0\text{s} - 9\text{s}$) to see each frame's thumbnail, classification, and confidence.
5. **Analyze Diagnostics:** Expand the bottom panel to inspect the confidence trajectory line chart and the root-cause dissonance report.
