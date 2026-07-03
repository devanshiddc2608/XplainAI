import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly
import json

from modules.model_utils import (
    logreg, rf, nn,
    X_test_scaled, X_test_unscaled,
    SCALED_FEATURES, UNSCALED_FEATURES
)

# ── Pre-create fast explainers at startup ─────────────────────────────────────
# LogReg: LinearExplainer is instant
explainer_lr = shap.Explainer(logreg, X_test_scaled)

# Random Forest: TreeExplainer is fast and exact
explainer_rf = shap.TreeExplainer(rf)

# Neural Network: KernelExplainer is slow — lazy-initialized on first use
_explainer_nn = None


def _get_nn_explainer():
    """Initialize NN explainer on first call only (takes ~5s to set up)."""
    global _explainer_nn
    if _explainer_nn is None:
        background = shap.sample(X_test_scaled, 50)
        _explainer_nn = shap.KernelExplainer(nn.predict, background)
    return _explainer_nn


def get_shap_contributions(model_name, df_scaled, df_unscaled, top_n=8):
    """
    Compute SHAP values for a single employee.
    Returns: (contributions list, base_value)
    contributions: list of (feature_name, shap_value, feature_raw_value)
    """
    if model_name == 'Logistic Regression':
        sv_obj = explainer_lr(df_scaled)
        sv = sv_obj.values[0]
        # Explainer auto-picks the right class for binary classifiers
        # If it returns 2D (both classes), take the positive class
        if sv.ndim == 2:
            sv = sv[:, 1]
        base_val = float(sv_obj.base_values[0])
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1])
        feature_names = SCALED_FEATURES
        feature_vals  = df_scaled.iloc[0].values

    elif model_name == 'Random Forest':
        sv_full = np.array(explainer_rf.shap_values(df_unscaled))
        # Shape: (n_samples, n_features, 2) — take class 1
        if sv_full.ndim == 3:
            sv = sv_full[0, :, 1]
        else:
            sv = sv_full[0]
        base_val = explainer_rf.expected_value
        if hasattr(base_val, '__len__'):
            base_val = float(base_val[1])
        feature_names = UNSCALED_FEATURES
        feature_vals  = df_unscaled.iloc[0].values

    elif model_name == 'Neural Network':
        explainer_nn = _get_nn_explainer()
        sv_raw = np.array(explainer_nn.shap_values(df_scaled))
        # Shape: (n_samples, n_features, 1) — squeeze last dim
        if sv_raw.ndim == 3:
            sv = sv_raw[0, :, 0]
        else:
            sv = sv_raw.flatten()
        base_val = explainer_nn.expected_value
        if hasattr(base_val, '__len__'):
            base_val = float(base_val[0])
        feature_names = SCALED_FEATURES
        feature_vals  = df_scaled.iloc[0].values

    # Top-N features by absolute SHAP value
    top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    contributions = [
        (feature_names[i], float(sv[i]), float(feature_vals[i]))
        for i in top_idx
    ]

    return contributions, float(base_val)


def create_waterfall_chart(contributions, base_value, final_prediction):
    """
    Build a horizontal Plotly waterfall chart from SHAP contributions.
    Returns JSON string for Plotly.js rendering in the browser.
    """
    # Sort by SHAP value (most negative at top, most positive at bottom)
    sorted_c = sorted(contributions, key=lambda x: x[1])

    y_labels = [f"{name}  (={val:.2f})" for name, sv, val in sorted_c]
    x_values = [sv for _, sv, _ in sorted_c]

    fig = go.Figure(go.Waterfall(
        orientation  = "h",
        measure      = ["relative"] * len(x_values) + ["total"],
        y            = y_labels + ["Final Risk Score"],
        x            = x_values + [final_prediction],
        base         = base_value,
        decreasing   = {"marker": {"color": "#22c55e"}},
        increasing   = {"marker": {"color": "#ef4444"}},
        totals       = {"marker": {"color": "#3b82f6"}},
        connector    = {"line": {"color": "#9ca3af", "width": 1}},
    ))

    fig.update_layout(
        height       = 420,
        margin       = dict(l=260, r=40, t=30, b=40),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#374151", size=12),
        xaxis_title   = "Impact on attrition probability",
        showlegend    = False,
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def compute_roc_data():
    """
    Compute ROC curve data for all three models at startup.
    Returns Plotly JSON string.
    """
    from sklearn.metrics import roc_curve, auc
    from modules.model_utils import y_test

    y_proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]
    y_proba_rf = rf.predict_proba(X_test_unscaled)[:, 1]
    y_proba_nn = nn.predict(X_test_scaled, verbose=0).flatten()

    fig = go.Figure()
    for name, y_prob, color in [
        ('Logistic Regression', y_proba_lr, '#3b82f6'),
        ('Random Forest',       y_proba_rf, '#10b981'),
        ('Neural Network',      y_proba_nn, '#f59e0b'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={roc_auc:.3f})",
            mode='lines',
            line=dict(color=color, width=2)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(color='#9ca3af', width=1, dash='dash')
    ))

    fig.update_layout(
        xaxis_title   = "False Positive Rate",
        yaxis_title   = "True Positive Rate",
        height        = 380,
        margin        = dict(l=50, r=30, t=30, b=50),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color="#374151"),
        legend        = dict(x=0.5, y=0.05),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)