import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    classification_report
)
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audit Framework Simulator",
    page_icon="⚖",
    layout="wide"
)

# ── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f8f8f8; }

    .audit-header {
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #e4e0dc;
        margin-bottom: 2rem;
    }
    .audit-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1410;
        letter-spacing: -0.02em;
    }
    .audit-subtitle {
        font-size: 0.95rem;
        color: #6d6560;
        margin-top: 0.4rem;
        line-height: 1.6;
    }

    .mode-card {
        background: white;
        border: 1px solid #e4e0dc;
        border-left: 3px solid #c8a030;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .mode-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1a1410;
    }
    .mode-desc {
        font-size: 0.8rem;
        color: #6d6560;
        margin-top: 0.2rem;
    }

    .result-pass {
        background: #f0f7f0;
        border: 1px solid #4a8a4a;
        border-left: 4px solid #4a8a4a;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .result-fail {
        background: #fdf5f0;
        border: 1px solid #8a4a30;
        border-left: 4px solid #8a4a30;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-rationale {
        font-size: 0.85rem;
        color: #3a3530;
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6d6560;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1410;
    }

    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #6d6560;
        font-weight: 700;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e4e0dc;
    }
</style>
""", unsafe_allow_html=True)


# ── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    df = df.dropna()
    df["income_binary"] = (df["income"].str.strip() == ">50K").astype(int)
    df["sex_binary"] = (df["sex"].str.strip() == "Male").astype(int)

    features = ["age", "education_num", "hours_per_week", "capital_gain",
                "capital_loss", "sex_binary"]
    X = df[features]
    y = df["income_binary"]
    sex = df["sex_binary"]
    return X, y, sex, features


# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_data
def train_model(model_type="Logistic Regression"):
    X, y, sex, features = load_data()
    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
        X, y, sex, test_size=0.3, random_state=42
    )
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, sex_test, y_prob


# ── GOVERNANCE MODES ─────────────────────────────────────────────────────────
GOVERNANCE_MODES = {
    "Regulatory-first": {
        "description": "Prioritizes discrimination minimization. Accepts lower accuracy to ensure demographic parity.",
        "threshold": 0.4,
        "parity_weight": 0.7,
        "accuracy_weight": 0.3,
        "fpr_tolerance": 0.05,
        "primary_metric": "demographic_parity",
        "pass_condition": "demographic_parity_gap < 0.1",
    },
    "Business-first": {
        "description": "Prioritizes predictive efficiency and accuracy. Accepts fairness tradeoffs for performance.",
        "threshold": 0.5,
        "parity_weight": 0.2,
        "accuracy_weight": 0.8,
        "fpr_tolerance": 0.15,
        "primary_metric": "accuracy",
        "pass_condition": "accuracy > 0.82",
    },
    "Risk-averse": {
        "description": "Minimizes false negatives. Prioritizes catching positive cases even at cost of false positives.",
        "threshold": 0.3,
        "parity_weight": 0.4,
        "accuracy_weight": 0.4,
        "fpr_tolerance": 0.2,
        "primary_metric": "false_negative_rate",
        "pass_condition": "false_negative_rate < 0.15",
    },
    "Public accountability": {
        "description": "Prioritizes explainability and equal opportunity across protected groups.",
        "threshold": 0.45,
        "parity_weight": 0.6,
        "accuracy_weight": 0.4,
        "fpr_tolerance": 0.08,
        "primary_metric": "equal_opportunity",
        "pass_condition": "equal_opportunity_gap < 0.08",
    },
}


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(y_test, y_prob, sex_test, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Demographic parity gap
    male_mask = sex_test == 1
    female_mask = sex_test == 0
    dp_male = y_pred[male_mask].mean() if male_mask.sum() > 0 else 0
    dp_female = y_pred[female_mask].mean() if female_mask.sum() > 0 else 0
    dp_gap = abs(dp_male - dp_female)

    # Equal opportunity gap (TPR difference)
    tpr_male = (y_pred[male_mask & (y_test == 1)].mean()
                if (male_mask & (y_test == 1)).sum() > 0 else 0)
    tpr_female = (y_pred[female_mask & (y_test == 1)].mean()
                  if (female_mask & (y_test == 1)).sum() > 0 else 0)
    eo_gap = abs(tpr_male - tpr_female)

    return {
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "dp_gap": dp_gap,
        "eo_gap": eo_gap,
        "dp_male": dp_male,
        "dp_female": dp_female,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def evaluate_audit(metrics, mode_config):
    pm = mode_config["primary_metric"]
    passed = False
    reason = ""

    if pm == "demographic_parity":
        passed = metrics["dp_gap"] < 0.1
        gap_pct = metrics["dp_gap"] * 100
        reason = (
            f"Demographic parity gap is {gap_pct:.1f}%. "
            f"{'Within' if passed else 'Exceeds'} the regulatory threshold of 10%. "
            f"Male positive rate: {metrics['dp_male']*100:.1f}%, "
            f"Female positive rate: {metrics['dp_female']*100:.1f}%."
        )
    elif pm == "accuracy":
        passed = metrics["accuracy"] > 0.82
        reason = (
            f"Model accuracy is {metrics['accuracy']*100:.1f}%. "
            f"{'Meets' if passed else 'Does not meet'} the business performance threshold of 82%. "
            f"Fairness considerations are secondary under this framework."
        )
    elif pm == "false_negative_rate":
        passed = metrics["fnr"] < 0.15
        reason = (
            f"False negative rate is {metrics['fnr']*100:.1f}%. "
            f"{'Within' if passed else 'Exceeds'} the risk tolerance of 15%. "
            f"This framework accepts higher false positives to minimize missed cases."
        )
    elif pm == "equal_opportunity":
        passed = metrics["eo_gap"] < 0.08
        reason = (
            f"Equal opportunity gap is {metrics['eo_gap']*100:.1f}%. "
            f"{'Within' if passed else 'Exceeds'} the public accountability threshold of 8%. "
            f"True positive rates are compared across gender groups."
        )

    return passed, reason


# ── CONFUSION MATRIX CHART ────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title=""):
    labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    values = cm.ravel()
    colors = ["#4a8a4a", "#8a4a30", "#8a4a30", "#4a8a4a"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside"
    ))
    fig.update_layout(
        title=title,
        height=280,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="#f8f8f8",
        font=dict(family="Inter", size=12, color="#3a3530"),
        showlegend=False,
    )
    return fig


# ── COMPARISON CHART ──────────────────────────────────────────────────────────
def plot_comparison(results_dict):
    modes = list(results_dict.keys())
    metrics_to_show = ["accuracy", "dp_gap", "eo_gap", "fnr"]
    metric_labels = ["Accuracy", "Demographic Parity Gap", "Equal Opportunity Gap", "False Negative Rate"]

    fig = go.Figure()
    colors = ["#c8a030", "#1c3557", "#4a8a4a", "#8a4a30"]

    for i, (metric, label) in enumerate(zip(metrics_to_show, metric_labels)):
        values = [results_dict[m][metric] for m in modes]
        fig.add_trace(go.Bar(
            name=label,
            x=modes,
            y=values,
            marker_color=colors[i],
            opacity=0.85,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="#f8f8f8",
        font=dict(family="Inter", size=11, color="#3a3530"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="audit-header">
        <div class="audit-title">Audit Framework Simulator</div>
        <div class="audit-subtitle">
            The same model can produce different audit outcomes depending on how evaluation
            frameworks define fairness, risk, and acceptable behavior.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data & model
    with st.spinner("Loading dataset and training model..."):
        model, X_test, y_test, sex_test, y_prob = train_model("Logistic Regression")

    # ── THREE COLUMN LAYOUT ──
    col_left, col_mid, col_right = st.columns([1, 1, 1.4])

    with col_left:
        st.markdown('<div class="section-label">Audit Regime</div>', unsafe_allow_html=True)

        selected_mode = st.selectbox(
            "Select regime",
            options=list(GOVERNANCE_MODES.keys()),
            label_visibility="collapsed"
        )

        mode_config = GOVERNANCE_MODES[selected_mode]

        for mode_name, cfg in GOVERNANCE_MODES.items():
            is_selected = mode_name == selected_mode
            border_color = "#c8a030" if is_selected else "#e4e0dc"
            bg = "rgba(200,160,48,0.05)" if is_selected else "white"
            text_color = "#1a1410" if is_selected else "#6d6560"
            st.markdown(f"""
            <div style="border:1px solid {border_color}; border-left:3px solid {border_color};
                        padding:.7rem .9rem; margin-bottom:.4rem; background:{bg};">
                <div style="font-size:0.8rem; font-weight:600; color:{text_color};">{mode_name}</div>
                <div style="font-size:0.72rem; color:#6d6560; margin-top:.2rem; line-height:1.5;">
                    {cfg['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Model</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem; color:#6d6560; line-height:1.8;">
            UCI Adult Income<br>
            Logistic Regression<br>
            Predict income &gt; $50K<br>
            Protected: Sex (M/F)
        </div>
        """, unsafe_allow_html=True)

        threshold = st.slider(
            "Decision threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(mode_config["threshold"]),
            step=0.05,
        )

        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Regulatory Context</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem; color:#6d6560; line-height:1.8;">
            Audit regime logic draws on evaluation principles embedded in:<br><br>
            <b style="color:#3a3530;">EU AI Act</b> — risk classification and conformity assessment obligations for high-risk systems<br><br>
            <b style="color:#3a3530;">GDPR Art. 22</b> — automated decision-making and non-discrimination requirements<br><br>
            <b style="color:#3a3530;">DSA (VLOP)</b> — transparency and systemic risk assessment for large platforms
        </div>
        """, unsafe_allow_html=True)

    metrics = compute_metrics(y_test, y_prob, sex_test, threshold)
    passed, rationale = evaluate_audit(metrics, mode_config)

    with col_mid:
        st.markdown('<div class="section-label">Audit Outcome</div>', unsafe_allow_html=True)

        result_color = "#4a8a4a" if passed else "#8a4a30"
        result_label = "PASSED" if passed else "FAILED"
        result_icon = "✓" if passed else "✗"

        st.markdown(f"""
        <div style="text-align:center; padding:2rem 1rem; border:1px solid {result_color};
                    background:{'rgba(74,138,74,0.05)' if passed else 'rgba(138,74,48,0.05)'};
                    margin-bottom:1.2rem;">
            <div style="font-size:2.8rem; color:{result_color}; font-weight:300;">{result_icon}</div>
            <div style="font-size:1.3rem; font-weight:700; color:{result_color};
                        letter-spacing:.1em;">{result_label}</div>
            <div style="font-size:0.75rem; color:#6d6560; margin-top:.4rem;
                        text-transform:uppercase; letter-spacing:.1em;">{selected_mode}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Metrics</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-label">Accuracy</div>'
                        f'<div class="metric-value">{metrics["accuracy"]:.3f}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label" style="margin-top:.6rem;">Parity Gap</div>'
                        f'<div class="metric-value">{metrics["dp_gap"]:.3f}</div>',
                        unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-label">Equal Opp. Gap</div>'
                        f'<div class="metric-value">{metrics["eo_gap"]:.3f}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label" style="margin-top:.6rem;">False Neg. Rate</div>'
                        f'<div class="metric-value">{metrics["fnr"]:.3f}</div>',
                        unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Audit Rationale</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding:1.4rem; border-left:3px solid {'#4a8a4a' if passed else '#8a4a30'};
                    background:white; min-height:140px;">
            <div style="font-family:'Georgia', serif; font-size:1.05rem; line-height:1.8;
                        color:#1a1410; font-style:italic;">
                "{result_icon} {result_label} under {selected_mode} framework."
            </div>
            <div style="font-size:0.85rem; color:#3a3530; margin-top:.8rem; line-height:1.75;">
                {rationale}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(
            plot_confusion_matrix(metrics["confusion_matrix"], ""),
            use_container_width=True
        )

    # ── COMPARISON ──
    st.markdown("---")
    st.markdown('<div class="section-label">All Regimes — Same Model</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.83rem; color:#6d6560; margin-bottom:1rem;">'
        'Each regime applies different institutional assumptions. '
        'Threshold values are set by regime priorities, not the model.</p>',
        unsafe_allow_html=True
    )

    all_results = {}
    all_pass_fail = {}
    for mode_name, cfg in GOVERNANCE_MODES.items():
        m = compute_metrics(y_test, y_prob, sex_test, cfg["threshold"])
        all_results[mode_name] = m
        p, r = evaluate_audit(m, cfg)
        all_pass_fail[mode_name] = (p, r)

    pf_cols = st.columns(4)
    for i, (mode_name, (p, r)) in enumerate(all_pass_fail.items()):
        with pf_cols[i]:
            color = "#4a8a4a" if p else "#8a4a30"
            label = "PASS" if p else "FAIL"
            st.markdown(f"""
            <div style="border:1px solid {color}; border-left:3px solid {color};
                        padding:.8rem; background:white;">
                <div style="font-size:0.7rem; text-transform:uppercase;
                            letter-spacing:.1em; color:#6d6560;">{mode_name}</div>
                <div style="font-size:1.1rem; font-weight:600; color:{color};
                            margin-top:.3rem;">{label}</div>
                <div style="font-size:0.72rem; color:#6d6560; margin-top:.3rem;
                            line-height:1.5;">{r[:80]}...</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_comparison(all_results), use_container_width=True)


if __name__ == "__main__":
    main()