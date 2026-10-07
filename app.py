import streamlit as st
import pandas as pd
import altair as alt
from video_processor import extract_frames_at_1fps
from verifier_engine import run_zero_shot_verification

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Scrap Verification & Provenance Engine",
    page_icon="♻️",
    layout="wide"
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
    .timeline-container {
        background-color: #111827;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #1F2937;
    }
    .timeline-cell {
        text-align: center;
        padding: 10px 4px;
        border-radius: 6px;
        font-family: monospace;
        margin: 2px;
    }
    .cell-hit {
        background-color: #064E3B;
        color: #6EE7B7;
        border: 1px solid #059669;
    }
    .cell-miss {
        background-color: #374151;
        color: #9CA3AF;
        border: 1px solid #4B5563;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("♻️ Material Verification & Provenance Engine")
st.caption("Temporal 1-Second Computer Vision Verification for Scrap Marketplace Intake")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Intake Settings")
    uploaded_file = st.file_uploader("Upload Scrap Video (10s Clip)", type=["mp4", "mov", "avi"])
    claimed_material = st.selectbox(
        "Seller Claimed Material",
        [
            "Copper (No. 1 Bare Bright)",
            "Aluminum (6063 Extrusions)",
            "Brass (Honey Scrap)",
            "Steel (HMS 1/2)"
        ]
    )
    threshold = st.slider("Verification Decision Threshold (τ)", 0.50, 0.95, 0.75, 0.05)
    st.info("System samples at 1.0 FPS across the 10-second inspection interval.")

# --- PROCESS VIDEO & INFERENCE ---
frames = []
video_name = "input_clip.mp4"
if uploaded_file is not None:
    video_name = uploaded_file.name
    frames = extract_frames_at_1fps(uploaded_file, max_seconds=10)

result = run_zero_shot_verification(
    frames=frames,
    claimed_material=claimed_material,
    threshold=threshold,
    video_name=video_name
)

# --- ROW 1: VIDEO PLAYER & DECISION PANEL ---
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.subheader(f"📹 {video_name} (10 s)")
    if uploaded_file is not None:
        st.video(uploaded_file)
    else:
        st.info("No video uploaded yet. Displaying baseline input stream preview:")
        st.video("https://www.w3schools.com/html/mov_bbb.mp4")

with col_right:
    st.subheader("📋 DECISION")
    
    badge_style = (
        "badge-strong" if result.decision_badge == "STRONG MATCH"
        else ("badge-moderate" if result.decision_badge == "MODERATE MATCH" else "badge-rejected")
    )
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 13px; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Verification Verdict</div>
        <div style="margin-top: 8px;">
            <span class="{badge_style}">[ {result.decision_badge} ]</span>
            <span style="font-size: 16px; font-weight: 600; margin-left: 10px;">({result.correspondence_score:.2f} correspondence)</span>
        </div>
        <div style="margin-top: 14px; font-size: 15px;">
            <b>Material:</b> <span style="color: #60A5FA; font-weight: 600;">{result.primary_material}</span> 
            <span style="color: #9CA3AF;">({result.primary_confidence:.2f})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("▼ Top-5 Probability Drawer", expanded=False):
        for mat_name, prob in result.top_5_probabilities.items():
            st.write(f"**{mat_name}**: `{prob*100:.1f}%`")
            st.progress(prob)

st.write("")

# --- ROW 2: PER-SECOND EVIDENCE (10-SECOND TEMPORAL LOCALIZATION) ---
st.subheader("⏱️ PER-SECOND EVIDENCE (10-Second Temporal Localization)")

timeline_cols = st.columns(10)
for idx, col in enumerate(timeline_cols):
    item = result.temporal_timeline[idx]
    cell_class = "cell-hit" if item.is_match else "cell-miss"
    symbol = "[██]" if item.is_match else "[  ]"
    
    with col:
        st.markdown(f"""
        <div class="timeline-cell {cell_class}">
            <div style="font-weight: bold; font-size: 12px;">{idx}s</div>
            <div style="font-size: 18px; margin: 4px 0;">{symbol}</div>
            <div style="font-size: 11px;">{item.confidence:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# --- ROW 3: EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS ---
with st.expander("🔍 EXPLAINABILITY & COUNTERFACTUAL DIAGNOSTICS", expanded=True):
    col_attr, col_chart = st.columns([1, 2], gap="large")
    
    with col_attr:
        st.markdown("#### Modality Attribution")
        st.metric(
            label="Visual Branch",
            value=f"{int(result.diagnostics.visual_branch_weight * 100)}%",
            delta="Primary Signal"
        )
        st.metric(
            label="Prior Metadata Branch",
            value=f"{int(result.diagnostics.metadata_branch_weight * 100)}%"
        )
        st.caption("Visual feature correspondence accounts for 86% of the final verification score.")

    with col_chart:
        st.markdown("#### Second-by-Second Cosine Similarity Trajectory")
        chart_df = pd.DataFrame({
            "Timestamp": [f"{i}s" for i in range(10)],
            "Cosine Similarity": result.diagnostics.trajectory,
            "Threshold (τ)": [threshold] * 10
        })

        line_chart = alt.Chart(chart_df).mark_line(point=True, color="#3B82F6").encode(
            x=alt.X("Timestamp:N", sort=None),
            y=alt.Y("Cosine Similarity:Q", scale=alt.Scale(domain=[0.0, 1.0])),
            tooltip=["Timestamp", "Cosine Similarity"]
        )
        threshold_rule = alt.Chart(chart_df).mark_rule(color="#EF4444", strokeDash=[4, 4]).encode(
            y="Threshold (τ):Q"
        )
        st.altair_chart(line_chart + threshold_rule, use_container_width=True)

    st.markdown("#### ⚡ Semantic Dissonance Diagnostic (Edge Case Explanations)")
    st.info(result.diagnostics.semantic_dissonance_notes)
