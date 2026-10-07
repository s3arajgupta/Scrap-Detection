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
        font-size: 14px;
        border: 1px solid #059669;
    }
    .badge-moderate {
        background-color: #78350F;
        color: #FBBF24;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 14px;
        border: 1px solid #D97706;
    }
    .badge-rejected {
        background-color: #7F1D1D;
        color: #F87171;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        font-size: 14px;
        border: 1px solid #DC2626;
    }
    .risk-badge {
        background-color: #1E293B;
        color: #94A3B8;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        border: 1px solid #334155;
        margin-left: 6px;
    }
    .action-box {
        background-color: #0F172A;
        border-left: 4px solid #10B981;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 10px;
        font-size: 13px;
        color: #E2E8F0;
    }
    .sample-box {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 6px;
        text-align: center;
        margin-bottom: 6px;
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
        font-size: 12px;
        color: #9CA3AF;
        margin-bottom: 2px;
    }
    .pred-label {
        font-size: 11px;
        font-weight: 600;
        color: #E5E7EB;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .conf-score {
        font-size: 12px;
        font-weight: 700;
        color: #60A5FA;
    }
    .margin-subtext {
        font-size: 10px;
        color: #9CA3AF;
        margin-top: 2px;
    }
    .policy-item {
        font-size: 13px;
        color: #CBD5E1;
        padding: 3px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER & VIDEO UPLOADER ---
st.title("♻️ Material Verification & Provenance Engine")
st.caption("Temporal Computer Vision Verification with Deep Explainability & Policy Transparency")

uploaded_file = st.file_uploader(
    "📤 **Upload Scrap Inspection Clip (10-Second MP4/MOV/AVI)**", 
    type=["mp4", "mov", "avi"],
    help="Samples 1 frame per second across 10 seconds and extracts visual evidence cues"
)

# --- INGEST & PROCESS VIDEO ---
frames = []
video_name = "input_clip.mp4"

if uploaded_file is not None:
    video_name = uploaded_file.name
    with st.spinner("Extracting 1 frame per second and computing visual evidence cues..."):
        frames = extract_frames_at_1fps(uploaded_file, max_seconds=10)

result, thumbnails = run_temporal_scrap_classification(
    frames=frames,
    video_name=video_name
)

st.divider()

# --- ROW 1: VIDEO PLAYER & DECISION TRANSPARENCY PANEL ---
col_video, col_decision = st.columns([1.1, 1], gap="large")

with col_video:
    st.subheader(f"📹 VIDEO PLAYER — {video_name} (10 s)")
    if uploaded_file is not None:
        st.video(uploaded_file)
    else:
        st.info("ℹ️ Upload a video above to test custom footage. Showing baseline preview stream:")
        st.video("https://www.w3schools.com/html/mov_bbb.mp4")

with col_decision:
    st.subheader("📋 DECISION & TRANSPARENCY POLICY")
    
    badge_style = (
        "badge-strong" if result.decision.decision_badge == "STRONG MATCH"
        else ("badge-moderate" if result.decision.decision_badge == "MODERATE MATCH" else "badge-rejected")
    )
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 11px; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Overall Verification Verdict</div>
        <div style="margin-top: 6px;">
            <span class="{badge_style}">[ {result.decision.decision_badge} ]</span>
            <span class="risk-badge">Rating: {result.decision.risk_rating}</span>
            <span style="font-size: 14px; font-weight: 600; margin-left: 8px;">({result.decision.persistence_percentage:.0f}% persistence)</span>
        </div>
        <div style="margin-top: 12px; font-size: 16px;">
            <b>Dominant Material:</b> <span style="color: #60A5FA; font-weight: 700;">{result.decision.dominant_material}</span>
        </div>
        <div style="margin-top: 4px; font-size: 12px; color: #9CA3AF;">
            <b>Temporal Agreement:</b> {result.decision.persistence_ratio} &nbsp;|&nbsp; 
            <b>Max Run:</b> {result.decision.longest_consecutive_run}s continuous &nbsp;|&nbsp; 
            <b>Mean Conf:</b> {result.decision.avg_confidence:.2f}
        </div>
        <div class="action-box">
            <b>⚡ Marketplace Action:</b> {result.decision.marketplace_action}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛡️ Decision Rationale & Policy Breakdown", expanded=False):
        st.markdown("**Automated Intake Rule Evaluation:**")
        for rationale in result.decision.decision_rationale:
            st.markdown(f"- {rationale}")

    with st.expander("▼ Top-5 Probability Distribution", expanded=False):
        for mat_name, prob in result.decision.top_5_aggregate.items():
            st.write(f"**{mat_name}**: `{prob*100:.1f}%`")
            st.progress(min(1.0, prob))

st.divider()

# --- ROW 2: PER-SECOND EVIDENCE (10 SAMPLE BOXES WITH THUMBNAILS & MARGINS) ---
st.subheader("⏱️ PER-SECOND EVIDENCE (10-Second Temporal Localization)")
st.caption("1 sample box for each 1-second frame. Shows predicted material, confidence, and margin over runner-up class.")

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
        <div style="text-align: center; margin-top: -6px;">
            <div class="pred-label" title="{sample.predicted_class}"><b>{short_label}</b></div>
            <div class="conf-score">{sample.confidence * 100:.0f}%</div>
            <div class="margin-subtext">Δ +{sample.confidence_margin*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# --- ROW 3: INTERACTIVE FRAME EXPLAINABILITY INSPECTOR ---
with st.expander("🔬 Frame-by-Frame Explainability & Visual Evidence Inspector", expanded=False):
    sel_sec = st.slider("Select 1-Second Sample Frame to Inspect Visual Cues", 0, 9, 0)
    selected_sample = result.samples[sel_sec]
    selected_thumb = thumbnails.get(sel_sec)
    
    col_f_img, col_f_details = st.columns([1, 2], gap="large")
    with col_f_img:
        if selected_thumb is not None:
            st.image(selected_thumb, caption=f"Frame at {sel_sec}s", width="stretch")
            
    with col_f_details:
        st.markdown(f"### Sample Frame #{sel_sec} ({sel_sec}s)")
        st.markdown(f"- **Primary Prediction:** `{selected_sample.predicted_class}` ({selected_sample.confidence*100:.1f}%)")
        st.markdown(f"- **Runner-Up Hypothesis:** `{selected_sample.runner_up_class}` ({selected_sample.runner_up_confidence*100:.1f}%)")
        st.markdown(f"- **Confidence Separation Margin:** `+{selected_sample.confidence_margin*100:.1f}%`")
        
        st.markdown("#### 🎯 Visual Evidence Driving this Prediction:")
        for cue in selected_sample.visual_evidence_cues:
            st.markdown(f"- {cue}")

st.divider()

# --- ROW 4: EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS ---
with st.expander("🔍 EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS", expanded=True):
    col_chart, col_notes = st.columns([1.4, 1.1], gap="large")
    
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

    st.markdown("#### 📊 Temporal Audit Trail & Frame-by-Frame Transparency Log")
    audit_data = []
    for s in result.samples:
        audit_data.append({
            "Second": f"{s.timestamp_sec}s",
            "Predicted Class": s.predicted_class,
            "Confidence": f"{s.confidence*100:.1f}%",
            "Dominant Agreement": "✅ Match" if s.is_dominant_match else "⚠️ Divergent",
            "Runner-Up Class": s.runner_up_class,
            "Runner-Up Conf": f"{s.runner_up_confidence*100:.1f}%",
            "Confidence Margin": f"+{s.confidence_margin*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(audit_data), use_container_width=False, width=1200)
