# Audit Framework Simulator

An interactive system examining how ML audit outcomes change depending on the institutional assumptions embedded within the evaluation framework itself.

## Core idea

The same model can produce different audit outcomes depending on how evaluation frameworks define fairness, risk, and acceptable behavior.

This project operationalizes the thesis concept of **structured epistemic dependence in ML auditing**: audit conclusions are not purely technical outputs, but are partially constructed through the evaluation framework itself.

## What this demonstrates

ML auditing relies on evaluation frameworks that embed institutional assumptions:

- **Labeling policies** — what counts as correct
- **Metric selection** — what gets measured
- **Threshold design** — what counts as compliant
- **Governance priorities** — whose interests are prioritized

These assumptions are not neutral. They are constructed within the same political and commercial environments as the systems they evaluate.

## Governance modes

| Mode | Primary objective | Threshold |
|------|------------------|-----------|
| Regulatory-first | Discrimination minimization | 0.40 |
| Business-first | Predictive efficiency | 0.50 |
| Risk-averse | Minimize false negatives | 0.30 |
| Public accountability | Equal opportunity across groups | 0.45 |

Each mode applies different fairness metrics and pass/fail criteria to the same model.

## Technical stack

- **Python** — core language
- **scikit-learn** — model training and evaluation
- **Streamlit** — interactive UI
- **Plotly** — visualization
- **Dataset** — UCI Adult Income (predict income > $50K)

## Setup

```bash
# Create environment
conda create -n audit-sim python=3.11
conda activate audit-sim

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Research context

This project is a computational artifact developed alongside MSc thesis research at emlyon Business School, examining structured epistemic dependence in AI auditing frameworks.

Drawing on Science and Technology Studies (STS) and critical algorithm studies, the research argues that the evaluation frameworks ML auditing relies on — labeling guidelines, performance metrics, and compliance thresholds — are institutionally constructed within the same political and commercial environments as the systems they assess.

Supervised by Saeed Varasteh Yazdi.

---

*Yu-Jou Ting · [everdinne.github.io](https://everdinne.github.io)*