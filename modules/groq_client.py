import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"


def _call_groq(system_prompt, user_prompt, temperature=0.3, max_tokens=600):
    """Centralized Groq call - all features route through here so error handling and
    model/version changes only need to happen in one place."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            return "The explanation could not be generated. Please try again."
        return text.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"
def explain_prediction(employee_data: dict, shap_contributions: list, risk_score: float, model_name: str) -> str:
    """
    employee_data: dict of feature_name -> value for this employee
    shap_contributions: list of (feature_name, shap_value) tuples, sorted by absolute impact, top ~6
    risk_score: float between 0 and 1
    model_name: which model produced this prediction
    """
    system_prompt = (
        "You are an HR analytics assistant explaining AI model predictions to HR managers "
        "who have no data science background. Write in plain, professional English. "
        "Never use technical terms like SHAP, coefficient, or feature importance - translate "
        "them into business language. Be specific and reference the actual factors given. "
        "Keep the response to one tight paragraph, 4-6 sentences."
    )

    contributions_text = "\n".join(
        [f"- {feat}: {'increases' if val > 0 else 'decreases'} risk (impact: {abs(val):.3f})"
         for feat, val in shap_contributions]
    )

    user_prompt = (
        f"Model used: {model_name}\n"
        f"Predicted attrition risk: {risk_score*100:.1f}%\n"
        f"Key factors driving this prediction:\n{contributions_text}\n\n"
        f"Explain in plain English why this employee has this risk level, and what it means "
        f"for an HR manager deciding whether to act."
    )

    return _call_groq(system_prompt, user_prompt, temperature=0.3, max_tokens=400)
def recommend_model(org_type: str, priority: str, comparison_summary: str) -> str:
    system_prompt = (
        "You are an AI governance consultant advising organizations on which machine learning "
        "model to deploy for employee attrition prediction, balancing accuracy against "
        "explainability and regulatory risk. Reference RBI's FREE-AI framework and EU AI Act "
        "high-risk system requirements where relevant to Indian or EU-exposed organizations. "
        "Give a clear, structured recommendation: state the recommended model first, then "
        "justify it in 3-4 sentences, then list deployment conditions as bullet points."
    )

    user_prompt = (
        f"Organisation type: {org_type}\n"
        f"Stated priority: {priority}\n"
        f"Model comparison data:\n{comparison_summary}\n\n"
        f"Recommend which model this organisation should deploy."
    )

    return _call_groq(system_prompt, user_prompt, temperature=0.4, max_tokens=500)
def generate_risk_assessment(model_name: str, org_type: str, regulatory_profile: str) -> str:
    system_prompt = (
        "You are a model risk and compliance officer producing a short regulatory risk "
        "assessment for an AI model being deployed in an organisation. Structure your response "
        "with these exact section headings: 'Regulatory Scrutiny Points', 'Required Documentation', "
        "'Monitoring Requirements', and 'Overall Risk Rating'. Be concise - 2-3 bullet points per "
        "section. Base your assessment only on the information given, do not invent regulations."
    )

    user_prompt = (
        f"Model: {model_name}\n"
        f"Organisation type: {org_type}\n"
        f"Model's regulatory profile: {regulatory_profile}\n\n"
        f"Generate the regulatory risk assessment."
    )

    return _call_groq(system_prompt, user_prompt, temperature=0.2, max_tokens=500)