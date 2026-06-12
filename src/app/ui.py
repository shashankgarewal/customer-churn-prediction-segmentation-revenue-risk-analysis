import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pathlib import Path
import yaml

from src.utils.common import get_project_root

root = get_project_root()
css_file = root / Path("src/app/style.css")

with open(css_file) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )
    
# pyrefly: ignore [missing-import]
from src.utils.config import API_BASE, SEGMENT_ORDER, BRAND_COLORS

st.set_page_config(
    page_title="Churn Intel",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def metric_card_html(label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'{sub_html}</div>')

def metric_row(*cards):
    st.markdown(f'<div class="metric-row">{"".join(cards)}</div>', unsafe_allow_html=True)

def section_header(num, title, desc_html, insight=None):
    insight_html = f'<div class="section-insight">💡 {insight}</div>' if insight else ""
    st.markdown(
        f'<div class="section-block">'
        f'<div class="section-num">{num}</div>'
        f'<div class="section-title">{title}</div>'
        f'<div class="section-desc">{desc_html}{insight_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def call_api(endpoint, files=None, n_samples=None, output="full"):
    try:
        if files:
            resp = requests.post(f"{API_BASE}{endpoint}", files=files, timeout=180)
        else:
            resp = requests.post(f"{API_BASE}{endpoint}",
                                 json={"n_samples": n_samples, "output": output}, timeout=180)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the server is running."
    except Exception as e:
        return None, str(e)

def get_request_params():
    if data_source == "Upload CSV":
        if uploaded_file is None:
            st.warning("Upload a CSV file first.")
            return None, None, False
        return None, uploaded_file, True
    return n_samples, None, True


PLOTLY_BASE   = dict(plot_bgcolor="#faf9f7", paper_bgcolor="#faf9f7",
                     font_color="#2a2520", font_family="Jost",
                     showlegend=True, xaxis_title=None)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📉 Churn <span>Intel</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-tagline">'
        'Churn Prediction · Persona Discovery<br>Retention Decision · Business Impact'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("**Data Source**")
    data_source = st.radio("ds", ["Use Test Dataset", "Upload CSV"],
                           label_visibility="collapsed")

    uploaded_file = None
    n_samples     = None

    if data_source == "Use Test Dataset":
        if st.checkbox("Random sample", value=False):
            n_samples = st.number_input("Rows", min_value=50, max_value=10000,
                                        value=500, step=50)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"],
                                         label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Run Inference**")
    run_model     = st.button("01 · Model Metrics")
    run_retention = st.button("02 · Retention Strategy")
    run_business  = st.button("03 · Business Impact")

    st.markdown("---")
    st.markdown("**Technical Details**")
    st.markdown("""
    <div class="sidebar-info">
    <b>Model</b><br>
    CatBoost · MLflow tracked<br>
    Optuna · 5 Studies · 2-phase tuning<br><br>x
    <b>Evaluation</b><br>
    Log Loss primary · AP + ROC per segment<br>
    Segment-specific thresholds<br><br>
    <b>Thresholds</b><br>
    VIP 0.18 · IMP 0.20<br>
    High 0.40 · Med 0.40 · Low 0.55<br><br>
    <b>Persona Discovery</b><br>
    SHAP positive drivers · Behavioral rules<br>
    Segment-aware retention playbook
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────

st.markdown("""
    <div class="page-header">
        <div class="byline">
            <div class="byline-left">
                Developed by 
                <a href="https://www.linkedin.com/in/shashankgarewal/" target="_blank" class="linkedin-link">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" class="linkedin-icon"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                    <span>Shashank Garewal</span>
                </a>
                &nbsp;·&nbsp; LTV-Segmented Churn Prediction &nbsp;·&nbsp;
                SHAP-Based Persona Discovery &nbsp;·&nbsp;
                Segment-Aware Retention Strategy  &nbsp;·&nbsp;
            </div>
            <div class="byline-right">
                <a href="https://github.com/shashankgarewal/customer-churn-prediction-segmentation-revenue-risk-analysis/" target="_blank" class="github-link">
                    <svg viewBox="0 0 16 16" width="16" height="16" class="github-icon" fill="currentColor"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82A7.443 7.443 0 0 0 8 3c-.73 0-1.44.1-2.11.29-1.53-1.04-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path></svg>
                    GitHub Project
                </a>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1> Churn Intelligence Engine</h1>
    <div class="project-desc">
        Predicting customer churn, determining customer persona through SHAP importance, recommending targeted retention actions, and identifying revenue risk.
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
            result, err = (call_api("/predict/upload", files={"file": f})
                           if f else
                           call_api("/predict/test", n_samples=n_s, output="full"))
        if err:   st.error(err)
        elif result: st.session_state["model_result"] = result

if "model_result" in st.session_state:
    result = st.session_state["model_result"]

    section_header(
        "01", "Model Performance",
        "<ul>"
        "<li>Selected CatBoost for final inference after utilizing Bayesian optimization to tune hyperparameters across five model families, minimizing negative log loss.</li>"
        "<li>Classify customer churn using segment-specific probability thresholds rather than a single global cutoff.</li>" 
        "<li>Each LTV tier operates at its own decision boundary - a VIP customer flagged at 45% churn probability carries more business weight than a Low-value customer at 55%, and the model is built to reflect that principal.</li>"
        "</ul>",
        insight="The model correctly identifies 8 in 10 customers likely to churn, with strongest accuracy on the highest-value segments."
    )

    if "error" in result:
        st.error(result["error"])

    elif "summary_by_segment" in result:
        summary = result["summary_by_segment"]
        metric_row(*[
            metric_card_html(seg,
                             f"{int(v['predicted_churners'])} / {int(v['total'])}",
                             f"avg prob {v['avg_churn_probability']:.3f}")
            for seg, v in summary.items()
        ])

    elif "metrics" in result:
        metrics = result["metrics"]
        overall = metrics.get("overall", {})
        per_seg = metrics.get("per_segment", {})

        total_customers = sum(v.get("total", 0) for v in per_seg.values())
        total_flagged   = sum(v.get("tp", 0) + v.get("fp", 0) for v in per_seg.values())

        metric_row(
            metric_card_html("Customers Evaluated", f"{total_customers:,}"),
            metric_card_html("Flagged Churners",    f"{total_flagged:,}"),
            metric_card_html("Overall Precision",   f"{overall.get('precision', 0):.3f}"),
            metric_card_html("Overall Recall",      f"{overall.get('recall', 0):.3f}"),
            metric_card_html("Overall F1",          f"{overall.get('f1', 0):.3f}"),
        )

        if per_seg:
            seg_rows = [
                {"Segment": seg, "Threshold": v.get("threshold"),
                 "Total": v.get("total"), "Precision": v.get("precision"),
                 "Recall": v.get("recall"), "F1": v.get("f1"),
                 "TP": v.get("tp"), "FP": v.get("fp"),
                 "FN": v.get("fn"), "TN": v.get("tn")}
                for seg, v in per_seg.items()
            ]
            df_seg = pd.DataFrame(seg_rows)

            fig1 = px.line(
                df_seg.melt(id_vars=["Segment"],
                            value_vars=["Precision", "Recall", "F1"],
                            var_name="Metric", value_name="Value"),
                x="Segment", y="Value", color="Metric", markers=True,
                category_orders={"Segment": SEGMENT_ORDER},
                color_discrete_map={"Precision": "#c0785a",
                                    "Recall": "#78a898", "F1": "#6b6259"}
            )
            fig1.update_traces(line=dict(width=3), marker=dict(size=10))
            fig1.update_layout(**PLOTLY_BASE, yaxis_range=[0, 1.05], yaxis_title="Score")
            fig1.update_xaxes(gridcolor="#c4baae")
            fig1.update_yaxes(gridcolor="#c4baae")
            st.plotly_chart(fig1, use_container_width=True)

            st.dataframe(df_seg.set_index("Segment"), width="stretch")

        if "predictions" in result:
            with st.expander(f"↓ Per-customer predictions ({len(result['predictions'])} rows)"):
                pred_df = pd.DataFrame(result["predictions"])
                cols    = [c for c in ["segment", "churn_probability",
                                       "predicted_churn", "actual_churn"]
                           if c in pred_df.columns]
                n_show  = st.slider("Rows", 10, min(500, len(pred_df)), 10, key="m_rows")
                st.dataframe(pred_df[cols].head(n_show), width="stretch")

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 02 — RETENTION STRATEGY
# ═══════════════════════════════════════════════════════════════
if run_retention:
    n_s, f, ok = get_request_params()
    if ok:
        with st.spinner("Running SHAP analysis and persona assignment..."):
            result, err = (call_api("/retention/upload", files={"file": f})
                           if f else
                           call_api("/retention/test", n_samples=n_s, output="full"))
        if err:   st.error(err)
        elif result: st.session_state["retention_result"] = result

if "retention_result" in st.session_state:
    result = st.session_state["retention_result"]

    section_header(
        "02", "Retention Strategy",
        "<ul>"
        "<li>For each predicted churner, positive SHAP contributions identify which behavioral signals are actively driving their churn risk. </li>"
        "<li>Such signals are matched against behavioral persona rules — Dormant, Disengaged, Frustrated, Price Sensitive — to profile the type of churn risk.</li>" 
        "<li>The segment and persona combination then maps to a targeted retention playbook with prioritized campaign actions.</li>"
        "</ul>",
        insight="Instead of a generic intervention, each at-risk customer receives a tailored action — a frustrated VIP gets account manager outreach, a price-sensitive medium customer gets a promotional campaign."
    )

    st.info(
        "**Return Rate vs. Service Calls — Two Distinct Behavioral Signals**\n\n"
        "Global SHAP analysis reveals an important distinction between these two features. "
        "Higher customer service calls correlate with **lower** churn probability — customers who reach out are actively engaged and less likely to leave quietly. "
        "Higher return rates correlate with **higher** churn probability, reflecting product dissatisfaction rather than engagement.\n\n"
        "Platforms that penalize high-return customers through fees or account restrictions risk "
        "accelerating churn in a group that is already structurally at risk. "
        "Any intervention triggered by return behavior must account for the reason behind returns, not just the rate."
    )

    if "error" in result:
        st.error(result["error"])
    elif "retention_summary" in result:
        summary     = result["retention_summary"]
        by_persona  = summary.get("by_persona",  {})
        by_priority = summary.get("by_priority", {})
        top_persona  = max(by_persona.items(),  key=lambda x: x[1], default=("—", 0))
        top_priority = max(by_priority.items(), key=lambda x: x[1], default=("—", 0))

        metric_row(
            metric_card_html("Predicted Churners",    f"{summary.get('total_churners', 0):,}"),
            metric_card_html("Avg Churn Probability", f"{summary.get('avg_probability', 0):.3f}"),
            metric_card_html("Top Persona",    top_persona[0],  f"{top_persona[1]} customers"),
            metric_card_html("Top Priority",   top_priority[0], f"{top_priority[1]} customers"),
        )

        if "churners" in result and result["churners"]:
            churn_df = pd.DataFrame(result["churners"])
            ct = (pd.crosstab(churn_df["segment"], churn_df["persona"])
                    .reindex(SEGMENT_ORDER).dropna(how="all").fillna(0))

            fig2 = px.bar(ct,
                          labels={"value": "Count", "segment": "Segment", "persona": "Persona"},
                          color_discrete_sequence=BRAND_COLORS)
            fig2.update_layout(**PLOTLY_BASE, barmode="stack", yaxis_title="Churners")
            fig2.update_xaxes(gridcolor="#c4baae")
            fig2.update_yaxes(gridcolor="#c4baae")
            st.plotly_chart(fig2, use_container_width=True)

        by_action = summary.get("by_action", {})
        if by_action:
            st.markdown("**Top Recommended Actions**")
            action_df = (pd.DataFrame(list(by_action.items()), columns=["Action", "Count"])
                         .sort_values("Count", ascending=False).head(10))
            st.dataframe(action_df, width="stretch", hide_index=True)

        if "churners" in result and result["churners"]:
            churners = result["churners"]
            with st.expander(f"↓ Per-churner detail ({len(churners)} customers)"):
                rows = []
                for c in churners:
                    d = c.get("top_churn_drivers",     [])
                    s = c.get("top_retention_signals", [])
                    a = c.get("retention_strategy", {}).get("recommended_actions", [])
                    rows.append({
                        "Segment":     c.get("segment"),
                        "Churn Prob":  c.get("churn_probability"),
                        "Persona":     c.get("persona"),
                        "2nd Persona": c.get("secondary_persona"),
                        "Priority":    c.get("retention_strategy", {}).get("priority"),
                        "Driver 1":    d[0]["feature"] if len(d) > 0 else "—",
                        "Driver 2":    d[1]["feature"] if len(d) > 1 else "—",
                        "Driver 3":    d[2]["feature"] if len(d) > 2 else "—",
                        "Signal 1":    s[0]["feature"] if len(s) > 0 else "—",
                        "Signal 2":    s[1]["feature"] if len(s) > 1 else "—",
                        "Actions":     " · ".join(a),
                    })
                churn_df   = pd.DataFrame(rows)
                core_cols  = ["Segment", "Churn Prob", "Persona", "Priority", "Actions"]
                extra_cols = ["2nd Persona", "Driver 1", "Driver 2", "Driver 3",
                              "Signal 1", "Signal 2"]
                n_show     = st.slider("Rows", 10, min(500, len(churn_df)), 10, key="r_rows")
                show_extra = st.checkbox("Show SHAP drivers and retention signals")
                st.dataframe(
                    churn_df[core_cols + (extra_cols if show_extra else [])].head(n_show),
                    width="stretch"
                )

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 03 — BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════
if run_business:
    n_s, f, ok = get_request_params()
    if ok:
        with st.spinner("Running business impact analysis..."):
            result, err = (call_api("/business/upload", files={"file": f})
                           if f else
                           call_api("/business/test", n_samples=n_s, output="full"))
        if err:   st.error(err)
        elif result: st.session_state["business_result"] = result

if "business_result" in st.session_state:
    result = st.session_state["business_result"]

    section_header(
        "03", "Business Impact",
        "<ul>"
        "<li>Financial exposure is calculated using customer Lifetime Value weighted by churn probability and model confidence.</li>"
        "<li>A predicted churners may still bring partial revenue even after leaving — probability-weighted risk makes a more honest exposure estimate than binary flags.</li>"
        "<li>Retention savings use segment-specific success rates — VIP and IMP receive higher-touch interventions with stronger expected outcomes, reflected in their higher estimate.</li>"
        "</ul>",
        insight="This section answers the core question: If we act on these predictions, how much revenue do we protect — and what are we leaving on the table if we don't?"
    )

    if "error" in result:
        st.error(result["error"])
    elif "business_impact" in result:
        per_seg = result["business_impact"].get("per_segment", {})

        total_exposure = sum(v.get("total_churn_exposure", 0) for v in per_seg.values())
        total_at_risk  = sum(v.get("revenue_at_risk", 0)      for v in per_seg.values())
        total_savings  = sum(v.get("retention_savings", 0)    for v in per_seg.values())
        total_missed   = sum(v.get("missed_revenue", 0)       for v in per_seg.values())
        avg_capture    = (sum(v.get("churn_capture_rate", 0) for v in per_seg.values()) / len(per_seg)
                          if per_seg else 0)

        metric_row(
            metric_card_html("Total Churn Exposure", f"${total_exposure:,.0f}",
                             "actual churner LTV pool"),
            metric_card_html("Revenue at Risk",      f"${total_at_risk:,.0f}",
                             "correctly identified churners"),
            metric_card_html("Retention Savings",    f"${total_savings:,.0f}",
                             "segment-specific success rates"),
            metric_card_html("Missed Revenue",       f"${total_missed:,.0f}",
                             "false negatives"),
            metric_card_html("Avg Capture Rate",     f"{avg_capture:.1%}",
                             "LTV-weighted across segments"),
        )

        if per_seg:
            seg_rows = [
                {"Segment":           seg,
                 "Retention Rate":    f"{v.get('retention_rate', 0):.0%}",
                 "Churners Found":    v.get("churners_identified"),
                 "Missed":            v.get("missed_churners"),
                 "Capture Rate":      f"{v.get('churn_capture_rate', 0):.1%}",
                 "Revenue at Risk":   f"${v.get('revenue_at_risk', 0):,.0f}",
                 "Retention Savings": f"${v.get('retention_savings', 0):,.0f}",
                 "Missed Revenue":    f"${v.get('missed_revenue', 0):,.0f}",
                 "Expected Risk":     f"${v.get('expected_financial_risk', 0):,.0f}",
                 "Total Exposure":    f"${v.get('total_churn_exposure', 0):,.0f}"}
                for seg, v in per_seg.items()
            ]
            st.dataframe(pd.DataFrame(seg_rows).set_index("Segment"), width="stretch")

            biz_data = [
                {"Segment":                    seg,
                 "Retention Savings":          per_seg[seg].get("retention_savings", 0),
                 "At Risk — Not Saved":        (per_seg[seg].get("revenue_at_risk", 0)
                                                - per_seg[seg].get("retention_savings", 0)),
                 "Missed Revenue":             per_seg[seg].get("missed_revenue", 0)}
                for seg in SEGMENT_ORDER if seg in per_seg
            ]
            if biz_data:
                fig3 = px.bar(
                    pd.DataFrame(biz_data), x="Segment",
                    y=["Retention Savings", "At Risk — Not Saved", "Missed Revenue"],
                    color_discrete_map={
                        "Retention Savings":   "#78a898",
                        "At Risk — Not Saved": "#c0785a",
                        "Missed Revenue":      "#b0a898",
                    }
                )
                fig3.update_layout(**PLOTLY_BASE, barmode="stack", yaxis_title="USD ($)")
                fig3.update_xaxes(gridcolor="#c4baae")
                fig3.update_yaxes(gridcolor="#c4baae")
                st.plotly_chart(fig3, use_container_width=True)

        if "predictions" in result:
            with st.expander(f"↓ Per-customer predictions ({len(result['predictions'])} rows)"):
                pred_df = pd.DataFrame(result["predictions"])
                cols    = [c for c in ["segment", "churn_probability",
                                       "predicted_churn", "actual_churn"]
                           if c in pred_df.columns]
                n_show  = st.slider("Rows", 10, min(500, len(pred_df)), 10, key="b_rows")
                st.dataframe(pred_df[cols].head(n_show), width="stretch")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="font-size:0.68rem; color:#b0a898; text-align:center; padding:0.4rem 0 1rem 0;">
    Churn Intelligence Engine &nbsp;·&nbsp;
    Built by <span style="color:#c0785a; font-weight:600;">Shashank Garewal</span>
    &nbsp;·&nbsp;
    CatBoost · SHAP · MLflow · Optuna · FastAPI · Streamlit
</div>
""", unsafe_allow_html=True)