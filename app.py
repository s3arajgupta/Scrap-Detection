import streamlit as st
import pandas as pd
import altair as alt
from video_processor import extract_frames_at_1fps
from verifier_engine import run_temporal_scrap_classification

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ScrapIQ - Physical Scrap Verification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INLINE CSS FOR INDUSTRIAL DASHBOARD STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1A1F2C;
        border-radius: 8px;
        padding: 18px 20px;
        border: 1px solid #2D3748;
        border-left: 5px solid #3B82F6;
        margin-bottom: 12px;
    }
    .badge-strong {
        background-color: #064E3B;
        color: #34D399;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 15px;
        border: 1px solid #059669;
    }
    .badge-moderate {
        background-color: #78350F;
        color: #FBBF24;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 15px;
        border: 1px solid #D97706;
    }
    .badge-rejected {
        background-color: #7F1D1D;
        color: #F87171;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 15px;
        border: 1px solid #DC2626;
    }
    .sample-box {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        margin-bottom: 8px;
    }
    .box-match {
        border-top: 4px solid #10B981;
    }
    .box-divergent {
        border-top: 4px solid #F59E0B;
        background-color: #1F2430;
    }
    .timestamp-header {
        font-weight: 700;
        font-size: 13px;
        color: #9CA3AF;
        margin-bottom: 4px;
    }
    .pred-label {
        font-size: 11px;
        font-weight: 600;
        color: #E5E7EB;
        margin-top: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .conf-score {
        font-size: 12px;
        font-weight: 700;
        color: #60A5FA;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER & VIDEO UPLOADER ---
st.title("♻️ Material Verification & Provenance Engine")
st.caption("Temporal 1-Second Computer Vision Classification (10-Sample Pretrained Model Analysis)")

uploaded_file = st.file_uploader(
    "📤 **Upload Scrap Video (10-Second Clip)**", 
    type=["mp4", "mov", "avi"],
    help="Upload an inspection clip to sample 1 frame per second and classify scrap material"
)

# --- INGEST & PROCESS VIDEO ---
frames = []
video_name = "input_clip.mp4"

if uploaded_file is not None:
    video_name = uploaded_file.name
    with st.spinner("Extracting 1 frame per second across 10 seconds..."):
        frames = extract_frames_at_1fps(uploaded_file, max_seconds=10)

result, thumbnails = run_temporal_scrap_classification(
    frames=frames,
    video_name=video_name
)

st.divider()

# --- ROW 1: VIDEO PLAYER & MAJORITY VOTING DECISION PANEL ---
col_video, col_decision = st.columns([1.1, 1], gap="large")

with col_video:
    st.subheader(f"📹 VIDEO PLAYER — {video_name} (10 s)")
    if uploaded_file is not None:
        st.video(uploaded_file)
    else:
        st.info("ℹ️ Upload a video above to test custom footage. Showing baseline preview stream:")
        st.video("https://www.w3schools.com/html/mov_bbb.mp4")

with col_decision:
    st.subheader("📋 DECISION (Majority Voting)")
    
    badge_style = (
        "badge-strong" if result.decision.decision_badge == "STRONG MATCH"
        else ("badge-moderate" if result.decision.decision_badge == "MODERATE MATCH" else "badge-rejected")
    )
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 12px; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Overall Verification Verdict</div>
        <div style="margin-top: 8px;">
            <span class="{badge_style}">[ {result.decision.decision_badge} ]</span>
            <span style="font-size: 15px; font-weight: 600; margin-left: 10px;">({result.decision.persistence_percentage:.0f}% persistence)</span>
        </div>
        <div style="margin-top: 14px; font-size: 16px;">
            <b>Dominant Material:</b> <span style="color: #60A5FA; font-weight: 700;">{result.decision.dominant_material}</span>
        </div>
        <div style="margin-top: 6px; font-size: 13px; color: #9CA3AF;">
            <b>Temporal Agreement:</b> {result.decision.persistence_ratio} &nbsp;|&nbsp; <b>Mean Confidence:</b> {result.decision.avg_confidence:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("▼ Top-5 Probability Drawer", expanded=False):
        for mat_name, prob in result.decision.top_5_aggregate.items():
            st.write(f"**{mat_name}**: `{prob*100:.1f}%`")
            st.progress(min(1.0, prob))

st.divider()

# --- ROW 2: PER-SECOND EVIDENCE (10 SAMPLE BOXES WITH THUMBNAILS) ---
st.subheader("⏱️ PER-SECOND EVIDENCE (10-Second Temporal Localization)")
st.caption("1 sample box for each 1-second frame. Green indicator shows agreement with the majority-voted dominant material.")

# Render 10 boxes in 10 responsive columns
sample_cols = st.columns(10)

for idx, col in enumerate(sample_cols):
    sample = result.samples[idx]
    box_class = "box-match" if sample.is_dominant_match else "box-divergent"
    thumb_img = thumbnails.get(idx)

    with col:
        st.markdown(f"""
        <div class="sample-box {box_class}">
            <div class="timestamp-header">{sample.timestamp_sec}s</div>
        </div>
        """, unsafe_allow_html=True)
        
        if thumb_img is not None:
            st.image(thumb_img, width="stretch")
            
        short_label = sample.predicted_class.replace(" scrap", "").replace(" contamination", "")
        st.markdown(f"""
        <div style="text-align: center; margin-top: -8px;">
            <div class="pred-label" title="{sample.predicted_class}"><b>{short_label}</b></div>
            <div class="conf-score">{sample.confidence * 100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- ROW 3: EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS ---
with st.expander("🔍 EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS", expanded=True):
    col_chart, col_notes = st.columns([1.5, 1], gap="large")
    
    with col_chart:
        st.markdown(f"#### Second-by-Second Trajectory ({result.decision.dominant_material})")
        chart_df = pd.DataFrame({
            "Timestamp": [f"{i}s" for i in range(10)],
            "Confidence": result.diagnostics.trajectory,
            "Baseline Threshold (0.75)": [0.75] * 10
        })

        line_chart = alt.Chart(chart_df).mark_line(point=True, color="#3B82F6").encode(
            x=alt.X("Timestamp:N", sort=None),
            y=alt.Y("Confidence:Q", scale=alt.Scale(domain=[0.0, 1.0]), title="Confidence Score"),
            tooltip=["Timestamp", "Confidence"]
        )
        threshold_rule = alt.Chart(chart_df).mark_rule(color="#EF4444", strokeDash=[4, 4]).encode(
            y="Baseline Threshold (0.75):Q"
        )
        st.altair_chart(line_chart + threshold_rule, width="stretch")

    with col_notes:
        st.markdown("#### ⚡ Semantic Dissonance Diagnostic")
        st.info(result.diagnostics.dissonance_explanation)
