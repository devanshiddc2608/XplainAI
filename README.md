# XplainAI — Model Explainability Gap Audit

> **Live App:** [xplainai-audit.onrender.com](https://xplainai-y1zg.onrender.com)
> *(Free tier — allow 30s on first load if cold-starting)*

---

## What This Project Does

Most ML portfolio projects build one model and show how accurate it is.

This project does something different: it builds **three models of increasing 
complexity** on the same HR attrition prediction problem, then systematically 
audits **what each model gains in accuracy versus what it loses in 
explainability** — concluding with an LLM-powered application where a 
business user can explore predictions, understand why the model made each 
decision in plain English, and receive a regulatory-framed deployment 
recommendation.

The core framing — accuracy versus explainability as a deliberate governance 
trade-off — is modelled on the kind of analysis a Big 4 firm or bank's Model 
Risk Management team would produce before deploying AI in a regulated context.

---

## Live Application

**[Open XplainAI →](https://xplainai-y1zg.onrender.com)**

| Page | What It Does |
|------|-------------|
| Audit Dashboard | 3-model comparison with interactive ROC curves and full audit table |
| Employee Analyser | Input any employee profile → get risk score + SHAP waterfall + plain-English explanation |
| Model Advisor | Describe your organisation → get LLM-generated deployment recommendation |
| Regulatory Assessment | Select model + org type → generate structured compliance document |

---

## The Explainability Gap Audit — Key Findings

Three models were trained on the IBM HR Analytics Employee Attrition dataset 
(1,470 employees, 44 features after encoding) with identical preprocessing:

| Model | Accuracy | F1 | ROC-AUC | Explainability | Regulatory Fit |
|-------|----------|-----|---------|----------------|----------------|
| Logistic Regression | 85.7% | 0.488 | 0.721 | ⭐⭐⭐⭐⭐ Native coefficients | ✅ High |
| Random Forest | 80.6% | 0.345 | 0.731 | ⭐⭐⭐⭐ SHAP TreeExplainer | ⚠️ Medium |
| Neural Network | 85.7% | 0.488 | 0.739 | ⭐⭐ SHAP KernelExplainer | ❌ Low |

**Audit conclusion:** Logistic Regression is recommended for regulated 
deployment. It matches the Neural Network's practical F1 performance while 
offering full native interpretability — no SHAP approximation, no stability 
risk. The Neural Network's SHAP explanations showed a stability score of 
0.0023 versus 0.0000 for Random Forest's exact TreeExplainer — over an order 
of magnitude less stable under small input perturbations, a meaningful 
auditability risk in employment decision contexts explicitly classified as 
high-risk under the EU AI Act.

This audit is framed against **RBI's FREE-AI Framework** (August 2025, 
"Understandable by Design" Sutra) and **EU AI Act** high-risk system 
transparency requirements.

---

## Sample Output

**Employee Risk Analysis (Random Forest)**

![SHAP Waterfall](assets/dashboard_screenshot.png)

---

## Technical Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.45-purple)
![Groq](https://img.shields.io/badge/Groq-Llama3.3--70b-green)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![Render](https://img.shields.io/badge/Deployed-Render-blue)

**ML:** scikit-learn, TensorFlow/Keras, imbalanced-learn (SMOTE)  
**Explainability:** SHAP (TreeExplainer, KernelExplainer, LinearExplainer)  
**LLM Layer:** Groq API (llama-3.3-70b-versatile)  
**Web:** Flask, Jinja2, Bootstrap 5, Plotly.js  
**Visualisation:** Plotly (interactive), Power BI (static audit dashboard)  
**Deployment:** Render, GitHub  

---

## Repository Structure

```
├── app.py                    # Flask application — routes and orchestration
├── modules/
│   ├── groq_client.py        # Groq API calls (3 features)
│   ├── model_utils.py        # Model loading, preprocessing, prediction
│   └── shap_utils.py         # SHAP computation, waterfall chart, ROC data
├── templates/                # Jinja2 HTML pages (4 pages + base)
├── notebooks/
│   ├── 01_data_prep.ipynb    # Data loading, SMOTE, encoding, scaling
│   ├── 02_models.ipynb       # Model training, SHAP plots, evaluation
│   ├── 03_audit.ipynb        # Comparison table, faithfulness/stability tests
│   └── 04_powerbi_export.ipynb # CSV exports for Power BI
├── model_cards/              # Model cards for all 3 models
├── models/                   # Saved model files
└── data/                     # Processed datasets + Power BI exports
```

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/xplainai-explainability-audit.git
cd xplainai-explainability-audit
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

```bash
python app.py
# Open http://localhost:5000
```

---

## Regulatory Context

This project references:
- **RBI FREE-AI Framework** (August 2025) — 7 Sutras for responsible AI in 
  Indian financial services, including "Understandable by Design"
- **EU AI Act** — employment decisions classified as high-risk AI requiring 
  transparency, human oversight, and audit trail documentation

---

*Built by Devanshi — [LinkedIn](https://www.linkedin.com/in/devanshi-chauhan-879031230/) · 
[Portfolio](YOUR_PORTFOLIO_URL)*
