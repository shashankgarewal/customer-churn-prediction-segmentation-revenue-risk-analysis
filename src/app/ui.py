import streamlit as st
import pandas as pd
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Ecom Churn Prediction & Retention Intelligence Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM DESIGN
# Lora italic for quotes/display, Jost for UI
# bg #faf9f7, accent #c0785a, dark #2a2520
# muted #b0a898, subtle #c4baae, warm-gray #6b6259
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400;1,600&family=Jost:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Jost', sans-serif !important;
    background-color: #faf9f7 !important;
    color: #2a2520 !important;
}

h1, h2, h3, h4 { font-family: 'Jost', sans-serif !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #2a2520 !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #faf9f7 !important; }
[data-testid="stSidebar"] .stRadio label { color: #b0a898 !important; font-size: 1.1rem !important; }
[data-testid="stSidebar"] .stCheckbox label { color: #b0a898 !important; font-size: 1.1rem !important; }

/* Page title area */
.page-title {
    border-bottom: 2px solid #c0785a;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
.page-title h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: #2a2520 !important;
    margin: 0 0 0.2rem 0 !important;
    line-height: 1.3 !important;
}
.page-title .byline {
    font-size: 1.0rem;
    color: #6b6259;
    font-weight: 400;
    letter-spacing: 0.04em;
}
.page-title .byline span {
    color: #c0785a;
    font-weight: 600;
}

/* Section header */
.section-block {
    background: #ffffff;
    border: 1px solid #c4baae;
    border-top: 3px solid #c0785a;
    border-radius: 3px;
    padding: 1.4rem 1.6rem;
    margin: 1.5rem 0 1rem 0;
}
.section-num {
    font-size: 0.65rem;
    font-weight: 700;
    color: #c0785a;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.section-title {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #2a2520 !important;
    margin: 0 0 0.6rem 0 !important;
}
.section-desc {
    font-family: 'Lora', serif !important;
    font-style: italic;
    font-size: 1.15rem;
    color: #6b6259;
    line-height: 1.65;
    margin: 0;
}

/* Metric cards */
.metric-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1.2rem 0; }
.metric-card {
    flex: 1;
    min-width: 180px;
    background: #ffffff;
    border: 1px solid #c4baae;
    border-left: 3px solid #c0785a;
    padding: 1rem 1.1rem;
    border-radius: 2px;
}
.metric-label {
    font-size: 1.0rem;
    font-weight: 600;
    color: #6b6259;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 2.6rem;
    font-weight: 700;
    color: #2a2520;
    line-height: 1.1;
}
.metric-sub {
    font-size: 1.0rem;
    color: #b0a898;
    margin-top: 0.25rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #c4baae;
    margin: 1.5rem 0;
}

/* Sidebar info block */
.sidebar-info {
    font-size: 0.85rem;
    color: #b0a898;
    line-height: 1.85;
}
.sidebar-info b { color: #c0785a; font-weight: 600; }
.sidebar-logo {
    font-size: 1.1rem;
    font-weight: 700;
    color: #faf9f7;
    margin-bottom: 0.2rem;
}
.sidebar-logo span { color: #c0785a; }
.sidebar-tagline {
    font-size: 0.68rem;
    color: #6b6259;
    margin-bottom: 1.2rem;
}

/* Priority badges */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 2px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.badge-critical { background: #f5e6e6; color: #8b2020; border: 1px solid #c0605060; }
.badge-high     { background: #f5ede0; color: #7a4010; border: 1px solid #c0785a60; }
.badge-medium   { background: #e8f0e8; color: #2a5a2a; border: 1px solid #4a8a4a60; }
.badge-low      { background: #e8eaf5; color: #2a2a6a; border: 1px solid #4a4a8a60; }
.badge-minimal  { background: #f0ede8; color: #6b6259; border: 1px solid #b0a89860; }

/* Tables */
[data-testid="stDataFrame"] { border: 1px solid #c4baae !important; }

/* Force brand colors on table headers and remove default streamlit grey */
[data-testid="stTable"] thead tr th, 
[data-testid="stDataFrame"] thead tr th,
[data-testid="stTable"] th,
div[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background-color: #f0ede8 !important; 
    color: #2a2520 !important; 
    font-weight: 600 !important;
    border-bottom: 2px solid #c4baae !important;
}

/* Buttons */
.stButton > button {
    background: #faf9f7 !important;
    color: #2a2520 !important;
    border: 1.5px solid #c0785a !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem 1rem !important;
    width: 100%;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #c0785a !important;
    color: #faf9f7 !important;
}

/* Sidebar button variant */
[data-testid="stSidebar"] .stButton > button {
    background: #2a2520 !important;
    color: #faf9f7 !important;
    border-color: #c0785a !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #c0785a !important;
    color: #2a2520 !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Jost', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #2a2520 !important;
    background: #f0ede8 !important;
    border: 1px solid #c4baae !important;
    border-radius: 2px !important;
    transition: all 0.2s ease;
}
.streamlit-expanderHeader:hover, .streamlit-expanderHeader[aria-expanded="true"] {
    background: #c0785a !important;
    color: #ffffff !important;
    border-color: #c0785a !important;
}
.streamlit-expanderHeader[aria-expanded="true"] svg {
    fill: #ffffff !important;
}

/* Radio and inputs in sidebar */
[data-testid="stSidebar"] .stNumberInput input {
    background: #1a1510 !important;
    color: #faf9f7 !important;
    border-color: #6b6259 !important;
    font-size: 0.82rem !important;
}

/* Warning/info */
.stAlert {
    background: #f5f0e8 !important;
    border: 1px solid #c4baae !important;
    border-left: 3px solid #c0785a !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def metric_card_html(label, value, sub=None):
    # Remove indentation from the string to prevent Streamlit from escaping it as a code block
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{sub_html}</div>'


def metric_row(*cards):
    inner = "".join(cards)
    st.markdown(f'<div class="metric-row">{inner}</div>', unsafe_allow_html=True)


def section_header(num, title, desc):
    st.markdown(f'<div class="section-block"><div class="section-num">{num}</div><div class="section-title">{title}</div><div class="section-desc">{desc}</div></div>', unsafe_allow_html=True)


def call_api(endpoint, files=None, n_samples=None, output="full"):
    try:
        if files:
            resp = requests.post(f"{API_BASE}{endpoint}", files=files, timeout=180)
        else:
            payload = {"n_samples": n_samples, "output": output}
            resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Run: `uvicorn src.app.api:app --reload`"
    except Exception as e:
        return None, str(e)


def get_request_params():
    if data_source == "Upload CSV":
        if uploaded_file is None:
            st.warning("Upload a CSV file first.")
            return None, None, False
        return None, uploaded_file, True
    return n_samples, None, True


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📉 Churn <span>Intel</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Prediction · Explainability · Retention · Business</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Data Source**")
    data_source = st.radio(
        label="data_source",
        options=["Use Test Dataset", "Upload CSV"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    n_samples = None

    if data_source == "Use Test Dataset":
        use_sample = st.checkbox("Random sample", value=False)
        if use_sample:
            n_samples = st.number_input(
                "Rows", min_value=50, max_value=10000, value=500, step=50
            )
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Run Inference**")
    run_model     = st.button("01 · Model Metrics")
    run_retention = st.button("02 · Retention Strategy")
    run_business  = st.button("03 · Business Impact")

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <b>Model</b><br>
    CatBoost · MLflow tracked<br>
    Optuna · 45 trials · 2-phase tuning<br>
    <br>
    <b>Evaluation</b><br>
    Log Loss primary · AP + ROC per segment<br>
    Segment-specific thresholds<br>
    <br>
    <b>Classification Thresholds</b><br>
    VIP 0.18 · IMP 0.20<br>
    High 0.40 · Med 0.40 · Low 0.55<br>
    <br>
    <b>Explainability</b><br>
    SHAP TreeExplainer · Local attribution<br>
    Persona rules · Retention playbook
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    <h1>Ecom Churn Intelligence Engine</h1>
    <div class="byline">
        Built by <span>Shashank Garewal</span> &nbsp;·&nbsp;
        LTV-Segmented Classification &nbsp;·&nbsp;
        SHAP-driven Contribution &nbsp;·&nbsp;
        Segment-Aware Retention Strategy
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 01 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════
if run_model:
    n_s, f, ok = get_request_params()
    if ok:
        with st.spinner("Running model inference..."):
            if f:
                result, err = call_api("/predict/upload", files={"file": f}, output="full")
            else:
                result, err = call_api("/predict/test", n_samples=n_s, output="full")
        if err:
            st.error(err)
        elif result:
            st.session_state["model_result"] = result

if "model_result" in st.session_state:
    result = st.session_state["model_result"]

    section_header(
        "01",
        "Model Performance",
        "CatBoost selected via Bayesian-optimized Optuna tuning across 5 model families. "
        "Evaluated using segment-specific probability thresholds — not a global 0.5 — "
        "to reflect real business operating points per LTV tier. "
        "Log Loss was the primary tuning metric; AP and ROC-AUC per segment drove final model selection."
    )

    if "error" in result:
        st.error(result["error"])

    elif "summary_by_segment" in result:
        # inference-only — no labels
        summary = result["summary_by_segment"]
        cards = [
            metric_card_html(seg,
                f"{int(v['predicted_churners'])} / {int(v['total'])}",
                f"avg prob {v['avg_churn_probability']:.3f}")
            for seg, v in summary.items()
        ]
        metric_row(*cards)

    elif "metrics" in result:
        metrics = result["metrics"]
        overall  = metrics.get("overall", {})
        per_seg  = metrics.get("per_segment", {})

        total_customers = sum(v.get("total", 0) for v in per_seg.values())
        total_flagged   = sum(v.get("tp", 0) + v.get("fp", 0) for v in per_seg.values())

        metric_row(
            metric_card_html("Customers Evaluated", f"{total_customers:,}"),
            metric_card_html("Flagged Churners", f"{total_flagged:,}"),
            metric_card_html("Overall Precision", f"{overall.get('precision', 0):.3f}"),
            metric_card_html("Overall Recall",    f"{overall.get('recall', 0):.3f}"),
            metric_card_html("Overall F1",        f"{overall.get('f1', 0):.3f}"),
        )

        if per_seg:
            seg_rows = []
            for seg, v in per_seg.items():
                seg_rows.append({
                    "Segment":   seg,
                    "Threshold": v.get("threshold"),
                    "Total":     v.get("total"),
                    "Precision": v.get("precision"),
                    "Recall":    v.get("recall"),
                    "F1":        v.get("f1"),
                    "TP": v.get("tp"), "FP": v.get("fp"),
                    "FN": v.get("fn"), "TN": v.get("tn"),
                })
            st.dataframe(
                pd.DataFrame(seg_rows).set_index("Segment"),
                width="stretch"
            )

        if "predictions" in result:
            with st.expander(f"↓ Per-customer predictions ({len(result['predictions'])} rows)"):
                pred_df = pd.DataFrame(result["predictions"])
                display_cols = [c for c in
                    ["segment", "churn_probability", "predicted_churn", "actual_churn"]
                    if c in pred_df.columns]
                n_show = st.slider("Rows", 10, min(500, len(pred_df)), 10, key="m_rows")
                st.dataframe(pred_df[display_cols].head(n_show), width="stretch")

st.markdown('<hr class="divider">', unsafe_allow_html=True)





# ═══════════════════════════════════════════════════════════════
# SECTION 02 — RETENTION STRATEGY
# ═══════════════════════════════════════════════════════════════
if run_retention:
    n_s, f, ok = get_request_params()
    if ok:
        with st.spinner("Running SHAP explanations and persona assignment..."):
            if f:
                result, err = call_api("/retention/upload", files={"file": f}, output="full")
            else:
                result, err = call_api("/retention/test", n_samples=n_s, output="full")
        if err:
            st.error(err)
        elif result:
            st.session_state["retention_result"] = result

if "retention_result" in st.session_state:
    result = st.session_state["retention_result"]

    section_header(
        "02",
        "Retention Strategy",
        "Predicted churners explained via SHAP TreeExplainer local feature attribution. "
        "Top positive SHAP contributors matched against behavioral persona rule sets — "
        "Dormant, Disengaged, Frustrated, Price Sensitive. "
        "Segment × persona combination determines intervention priority and recommended campaign actions."
    )

    if "error" in result:
        st.error(result["error"])
    elif "retention_summary" in result:
        summary = result["retention_summary"]

        by_persona  = summary.get("by_persona",  {})
        by_priority = summary.get("by_priority", {})
        top_persona  = max(by_persona.items(),  key=lambda x: x[1], default=("—", 0))
        top_priority = max(by_priority.items(), key=lambda x: x[1], default=("—", 0))

        metric_row(
            metric_card_html("Predicted Churners",  f"{summary.get('total_churners', 0):,}"),
            metric_card_html("Avg Churn Probability", f"{summary.get('avg_probability', 0):.3f}"),
            metric_card_html("Top Persona",   top_persona[0],  f"{top_persona[1]} customers"),
            metric_card_html("Top Priority",  top_priority[0], f"{top_priority[1]} customers"),
        )

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**By Persona**")
            if by_persona:
                persona_df = (
                    pd.DataFrame(list(by_persona.items()), columns=["Persona", "Count"])
                    .sort_values("Count", ascending=False)
                )
                st.dataframe(persona_df, width="stretch", hide_index=True)

        with col_r:
            st.markdown("**By Priority**")
            if by_priority:
                priority_order = ["Critical", "High", "Medium", "Low", "Minimal"]
                priority_df = pd.DataFrame(
                    list(by_priority.items()), columns=["Priority", "Count"]
                )
                priority_df["_ord"] = priority_df["Priority"].apply(
                    lambda x: priority_order.index(x) if x in priority_order else 99
                )
                priority_df = priority_df.sort_values("_ord").drop(columns="_ord")
                st.dataframe(priority_df, width="stretch", hide_index=True)
        
        # top actions
        st.markdown("**Top Recommended Actions**")
        by_action = summary.get("by_action", {})
        if by_action:
            action_df = (
                pd.DataFrame(list(by_action.items()), columns=["Action", "Count"])
                .sort_values("Count", ascending=False)
                .head(10)
            )
            st.dataframe(action_df, width="stretch", hide_index=True)
        # per-churner detail table
        if "churners" in result and result["churners"]:
            churners = result["churners"]

            with st.expander(f"↓ Per-churner detail ({len(churners)} customers)"):
                rows = []
                for c in churners:
                    drivers = c.get("top_churn_drivers",     [])
                    signals = c.get("top_retention_signals", [])
                    actions = c.get("retention_strategy", {}).get("recommended_actions", [])
                    rows.append({
                        "Segment":      c.get("segment"),
                        "Churn Prob":   c.get("churn_probability"),
                        "Persona":      c.get("persona"),
                        "2nd Persona":  c.get("secondary_persona"),
                        "Priority":     c.get("retention_strategy", {}).get("priority"),
                        "Driver 1":     drivers[0]["feature"] if len(drivers) > 0 else "—",
                        "Driver 2":     drivers[1]["feature"] if len(drivers) > 1 else "—",
                        "Driver 3":     drivers[2]["feature"] if len(drivers) > 2 else "—",
                        "Signal 1":     signals[0]["feature"] if len(signals) > 0 else "—",
                        "Signal 2":     signals[1]["feature"] if len(signals) > 1 else "—",
                        "Actions":      " · ".join(actions),
                    })

                churn_df = pd.DataFrame(rows)
                core_cols  = ["Segment", "Churn Prob", "Persona", "Priority", "Actions"]
                extra_cols = ["2nd Persona", "Driver 1", "Driver 2", "Driver 3",
                              "Signal 1", "Signal 2"]

                n_show     = st.slider("Rows", 10, min(500, len(churn_df)), 10, key="r_rows")
                show_extra = st.checkbox("Show SHAP drivers & retention signals")
                display_cols = core_cols + (extra_cols if show_extra else [])

                st.dataframe(
                    churn_df[display_cols].head(n_show),
                    width="stretch"
                )


# ═══════════════════════════════════════════════════════════════
# SECTION 03 — BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════
if run_business:
    n_s, f, ok = get_request_params()
    if ok:
        with st.spinner("Running business impact analysis..."):
            if f:
                result, err = call_api("/business/upload", files={"file": f}, output="full")
            else:
                result, err = call_api("/business/test", n_samples=n_s, output="full")
        if err:
            st.error(err)
        elif result:
            st.session_state["business_result"] = result

if "business_result" in st.session_state:
    result = st.session_state["business_result"]

    section_header(
        "03",
        "Business Impact",
        "Impact computed against ground-truth labels using LTV-weighted metrics across all LTV segments. "
        "Retention savings estimated using segment-specific success rates — VIP/IMP receive higher-touch "
        "interventions with stronger retention likelihood. Expected financial risk weights predicted churn "
        "probability against LTV, incorporating model confidence directly into the business exposure figure."
    )

    if "error" in result:
        st.error(result["error"])
    elif "business_impact" in result:
        per_seg = result["business_impact"].get("per_segment", {})

        total_exposure = sum(v.get("total_churn_exposure", 0) for v in per_seg.values())
        total_at_risk  = sum(v.get("revenue_at_risk", 0)      for v in per_seg.values())
        total_savings  = sum(v.get("retention_savings", 0)    for v in per_seg.values())
        total_missed   = sum(v.get("missed_revenue", 0)       for v in per_seg.values())
        avg_capture    = (
            sum(v.get("churn_capture_rate", 0) for v in per_seg.values()) / len(per_seg)
            if per_seg else 0
        )

        metric_row(
            metric_card_html("Total Churn Exposure", f"${total_exposure:,.0f}",
                             "actual churner LTV pool"),
            metric_card_html("Revenue at Risk",      f"${total_at_risk:,.0f}",
                             "correctly identified churners"),
            metric_card_html("Retention Savings",    f"${total_savings:,.0f}",
                             "@ 70% success rate"),
            metric_card_html("Missed Revenue",       f"${total_missed:,.0f}",
                             "false negatives"),
            metric_card_html("Avg Capture Rate",     f"{avg_capture:.1%}",
                             "LTV-weighted across segments"),
        )

        if per_seg:
            seg_rows = []
            for seg, v in per_seg.items():
                seg_rows.append({
                    "Segment":           seg,
                    "Retention Rate":    f"{v.get('retention_rate', 0):.0%}",
                    "Churners Found":    v.get("churners_identified"),
                    "Missed":            v.get("missed_churners"),
                    "Capture Rate":      f"{v.get('churn_capture_rate', 0):.1%}",
                    "Revenue at Risk":   f"${v.get('revenue_at_risk', 0):,.0f}",
                    "Retention Savings": f"${v.get('retention_savings', 0):,.0f}",
                    "Missed Revenue":    f"${v.get('missed_revenue', 0):,.0f}",
                    "Expected Risk":     f"${v.get('expected_financial_risk', 0):,.0f}",
                    "Total Exposure":    f"${v.get('total_churn_exposure', 0):,.0f}",
                })
            st.dataframe(
                pd.DataFrame(seg_rows).set_index("Segment"),
                width="stretch"
            )

        # per-customer detail
        if "predictions" in result:
            with st.expander(f"↓ Per-customer predictions ({len(result['predictions'])} rows)"):
                pred_df = pd.DataFrame(result["predictions"])
                display_cols = [c for c in
                    ["segment", "churn_probability", "predicted_churn", "actual_churn"]
                    if c in pred_df.columns]
                n_show = st.slider("Rows", 10, min(500, len(pred_df)), 10, key="b_rows")
                st.dataframe(pred_df[display_cols].head(n_show), width="stretch")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="font-size:0.7rem; color:#b0a898; text-align:center; padding: 0.5rem 0 1rem 0;">
    Ecom Churn Prediction &amp; Retention Intelligence Engine &nbsp;·&nbsp;
    Built by <span style="color:#c0785a; font-weight:600;">Shashank Garewal</span> &nbsp;·&nbsp;
    CatBoost · SHAP · MLflow · Optuna · FastAPI · Streamlit
</div>
""", unsafe_allow_html=True)