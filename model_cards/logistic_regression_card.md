# Model Card — Logistic Regression (Attrition Risk Scorer)

## Model Description
A binary logistic regression classifier trained to predict employee attrition 
probability. Produces a probability score between 0 and 1 representing 
likelihood of voluntary departure within the next review period.

- **Model type:** Logistic Regression (scikit-learn, max_iter=1000)
- **Training framework:** scikit-learn 1.9.0
- **Input:** 44 features (after one-hot encoding of 7 categorical variables)
- **Output:** Probability of attrition (class 1)

## Intended Use
- **Primary use:** HR analytics — identifying employees at elevated attrition 
  risk for proactive retention intervention
- **Primary users:** HR Business Partners, People Analytics teams
- **Deployment context:** Decision-support tool only; all flagged cases require 
  human review before any action is taken

## Out of Scope Uses
- Automated termination or disciplinary decisions
- Performance evaluation
- Hiring or promotion decisions
- Any deployment without human oversight
- Deployment on populations substantially different from IBM's HR dataset 
  demographic profile

## Training Data
- **Dataset:** IBM HR Analytics Employee Attrition dataset (synthetic/illustrative)
- **Size:** 1,470 employees, 35 original features
- **Class distribution:** 83.9% No Attrition, 16.1% Attrition (imbalanced)
- **Imbalance handling:** SMOTE applied to training set only (not test set)
- **Train/test split:** 80/20 stratified split (random_state=42)
- **Preprocessing:** StandardScaler applied; categorical variables one-hot 
  encoded with drop_first=True

## Evaluation Results
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.857  |
| Precision | 0.571  |
| Recall    | 0.426  |
| F1 Score  | 0.488  |
| ROC-AUC   | 0.721  |

**Confusion matrix (test set, n=294):**
- True Negatives: 232 | False Positives: 15
- False Negatives: 27  | True Positives: 20

## Explainability
- **Method:** Native coefficients (interpretable by design)
- **Quality rating:** 5/5 — explanation IS the model; no approximation risk
- **Top attrition risk factor:** OverTime_Yes (odds ratio 1.93)
- **Top protective factor:** Department_Research & Development (odds ratio 0.19)
- **Regulatory suitability:** HIGH — fully auditable under RBI FREE-AI 
  "Understandable by Design" Sutra; compatible with EU AI Act high-risk 
  transparency requirements for employment decisions

## Ethical Considerations
- Dataset is IBM's synthetic illustrative data — real-world bias audits 
  across gender, age, and marital status are required before production use
- OverTime and MaritalStatus features may act as proxies for protected 
  characteristics in some jurisdictions
- Model should never be sole input to any HR decision affecting an individual

## Caveats and Recommendations
- Retrain if organizational attrition base rate shifts by more than 5 
  percentage points
- Monitor for prediction drift quarterly
- SHAP coefficient explanations must accompany every individual prediction 
  surfaced to an HR manager
- This model is trained on synthetic data; validation on real organizational 
  data is required before any production deployment