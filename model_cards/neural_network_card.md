# Model Card — Neural Network (Attrition Risk Scorer)

## Model Description
A feedforward neural network with 2 hidden layers trained to predict 
employee attrition probability.

- **Architecture:** Input(44) → Dense(32, ReLU) → Dropout(0.3) → 
  Dense(16, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- **Total parameters:** 1,985
- **Training framework:** TensorFlow/Keras 2.16.2
- **Optimizer:** Adam | **Loss:** Binary cross-entropy
- **Early stopping:** patience=10 on validation loss, 
  restore_best_weights=True
- **Random seeds:** PYTHONHASHSEED=42, numpy seed=42, tf seed=42

## Intended Use
Same as Logistic Regression card — decision-support only, human review 
required. Given lower explainability quality, human review is especially 
critical for this model.

## Out of Scope Uses
Same as Logistic Regression card, plus:
- Any deployment where explanation of individual decisions is legally required 
  without human intermediary
- Any context where SHAP approximation instability is unacceptable

## Training Data
Same dataset and preprocessing as Logistic Regression (StandardScaler applied).

## Evaluation Results
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.857  |
| Precision | 0.571  |
| Recall    | 0.426  |
| F1 Score  | 0.488  |
| ROC-AUC   | 0.739  |

**Note:** Highest ROC-AUC of the three models (0.739 vs 0.721 for LR) 
but identical F1/precision/recall to Logistic Regression at 0.5 threshold.

## Explainability
- **Method:** SHAP KernelExplainer (sampling-based approximation)
- **Quality rating:** 2/5 — approximation method; run twice on same input, 
  get slightly different explanations
- **SHAP faithfulness score:** 100% (n=20; treat with caution — small sample, 
  not proof of reliable faithfulness at scale)
- **SHAP stability score:** 0.0023 mean diff under small perturbations — 
  over an order of magnitude less stable than Random Forest's TreeExplainer
- **Regulatory suitability:** LOW — KernelExplainer approximation creates 
  auditability risk; explanation reproducibility cannot be guaranteed; 
  not recommended for regulated deployment without human override on 
  every flagged case

## Ethical Considerations
Same as Logistic Regression card. The reduced explainability quality means 
biases are harder to detect and audit — heightening the ethical obligation 
for human review.

## Caveats and Recommendations
- Never deploy without mandatory human review of every prediction
- SHAP explanation instability (0.0023) means two auditors running the 
  explanation tool on the same employee may see different top factors
- Recommended only in low-regulatory-risk contexts (e.g. internal HR 
  analytics dashboard with no automated decision output)
- KernelExplainer computation time (~60s per employee) makes real-time 
  deployment impractical without pre-computation