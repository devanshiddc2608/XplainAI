from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

# ── Load comparison data ───────────────────────────────────────────────────────
comparison_df = pd.read_csv(BASE_DIR / 'data' / 'model_comparison.csv')

# ── Pre-compute ROC curves once at startup ────────────────────────────────────
# Imported here so models are already loaded by model_utils
from modules.shap_utils import compute_roc_data, get_shap_contributions, create_waterfall_chart
from modules.model_utils import predict, CATEGORY_VALUES
from modules.groq_client import explain_prediction, recommend_model, generate_risk_assessment

ROC_CHART_JSON = compute_roc_data()


# ── Page 1 — Audit Dashboard ──────────────────────────────────────────────────
@app.route('/')
def index():
    comparison = comparison_df.to_dict('records')
    return render_template('index.html',
                           comparison=comparison,
                           roc_chart=ROC_CHART_JSON)


# ── Page 2 — Employee Risk Analyser ───────────────────────────────────────────
@app.route('/analyser')
def analyser():
    return render_template('analyser.html', categories=CATEGORY_VALUES)


@app.route('/analyse', methods=['POST'])
def analyse():
    try:
        model_name = request.form.get('model_name', 'Random Forest')

        # Prediction
        risk_score, df_scaled, df_unscaled = predict(model_name, request.form)

        # SHAP
        contributions, base_value = get_shap_contributions(
            model_name, df_scaled, df_unscaled
        )

        # Waterfall chart
        chart_json = create_waterfall_chart(contributions, base_value, risk_score)

        # Groq plain-English explanation
        shap_for_groq = [(name, sv) for name, sv, _ in contributions]
        explanation   = explain_prediction(
            employee_data     = dict(request.form),
            shap_contributions = shap_for_groq,
            risk_score        = risk_score,
            model_name        = model_name
        )

        # Risk level band
        risk_level = 'low' if risk_score < 0.3 else ('high' if risk_score >= 0.6 else 'medium')

        return jsonify({
            'success':    True,
            'risk_score': round(risk_score * 100, 1),
            'risk_level': risk_level,
            'chart':      chart_json,
            'explanation': explanation,
            'model_name': model_name,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Page 3 — Model Selection Advisor ─────────────────────────────────────────
@app.route('/advisor')
def advisor():
    return render_template('advisor.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        org_type = request.form.get('org_type', 'Large Corporate HR Department')
        priority = request.form.get('priority', 'Balanced Approach')

        cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1',
                'ROC-AUC', 'Explainability Method',
                'Explainability Quality (1-5)', 'Regulatory Suitability']
        summary = comparison_df[cols].to_string(index=False)

        recommendation = recommend_model(org_type, priority, summary)

        return jsonify({'success': True, 'recommendation': recommendation})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Page 4 — Regulatory Risk Assessment ──────────────────────────────────────
@app.route('/assessment')
def assessment():
    models = comparison_df['Model'].tolist()
    return render_template('assessment.html', models=models)


@app.route('/generate_assessment', methods=['POST'])
def generate_assessment_route():
    try:
        model_name = request.form.get('model_name', 'Random Forest')
        org_type   = request.form.get('org_type', 'Regulated Bank')

        row = comparison_df[comparison_df['Model'] == model_name].iloc[0]
        regulatory_profile = (
            f"Explainability method: {row['Explainability Method']}. "
            f"Explainability quality score: {row['Explainability Quality (1-5)']}/5. "
            f"Regulatory suitability: {row['Regulatory Suitability']}. "
            f"Computational cost: {row['Computational Cost']}. "
            f"Maintenance complexity: {row['Maintenance Complexity']}. "
            f"Test set ROC-AUC: {row['ROC-AUC']}, F1: {row['F1']}."
        )

        assessment_text = generate_risk_assessment(model_name, org_type, regulatory_profile)

        return jsonify({
            'success':    True,
            'assessment': assessment_text,
            'model_name': model_name,
            'org_type':   org_type,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)