from modules.groq_client import explain_prediction, recommend_model, generate_risk_assessment

result = explain_prediction(
    employee_data={},
    shap_contributions=[("OverTime", 0.18), ("MonthlyIncome", -0.12), ("JobSatisfaction", -0.09)],
    risk_score=0.72,
    model_name="Random Forest"
)
print("FEATURE 1 OUTPUT:\n", result)
# In test_groq.py — extend it with these three calls
from modules.groq_client import explain_prediction, recommend_model, generate_risk_assessment

# Test 1: High-risk explanation
print(explain_prediction(
    employee_data={},
    shap_contributions=[("OverTime", 0.18), ("StockOptionLevel", -0.12),
                        ("JobSatisfaction", -0.09), ("YearsAtCompany", 0.07)],
    risk_score=0.78,
    model_name="Random Forest"
))

# Test 2: Low-risk explanation
print(explain_prediction(
    employee_data={},
    shap_contributions=[("StockOptionLevel", -0.15), ("MaritalStatus_Married", -0.08),
                        ("JobSatisfaction", -0.07)],
    risk_score=0.12,
    model_name="Logistic Regression"
))

# Test 3: Recommendation
print(recommend_model(
    org_type="Regulated Bank",
    priority="Full Regulatory Compliance",
    comparison_summary="LR: F1=0.488, AUC=0.721, Explainability=5/5\nRF: F1=0.345, AUC=0.731, Explainability=4/5\nNN: F1=0.488, AUC=0.739, Explainability=2/5"
))

# Test 4: Assessment
print(generate_risk_assessment(
    model_name="Neural Network",
    org_type="Regulated Bank",
    regulatory_profile="Explainability method: SHAP KernelExplainer. Quality: 2/5. Suitability: Low."
))