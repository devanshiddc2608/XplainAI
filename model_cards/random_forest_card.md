# Model Card — Random Forest (Attrition Risk Scorer)

## Model Description
A Random Forest ensemble classifier trained to predict employee attrition 
probability using 200 decision trees.

- **Model type:** RandomForestClassifier 
  (n_estimators=200, max_depth=8, min_samples_leaf=5)
- **Training framework:** scikit-learn 1.9.0
- **Input:** 44 features (unscaled — tree-based models do not require scaling)
- **Output:** Probability of attrition (class 1)

## Intended Use
Same as Logistic Regression card — decision-support only, human review required.

## Out of Scope Uses
Same as Logistic Regression card.

## Training Data
Same dataset and preprocessing pipeline as Logistic Regression, except 
StandardScaler is NOT applied (tree-based models are scale-invariant).

## Evaluation Results
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.806  |
| Precision | 0.375  |
| Recall    | 0.319  |
| F1 Score  | 0.345  |
| ROC-AUC   | 0.731  |

**Note:** Underperforms Logistic Regression on all threshold-based metrics 
despite comparable AUC, likely due to conservative hyperparameters 
(max_depth=8) constraining the model's capacity on this small dataset.

## Explainability
- **Method:** SHAP TreeExplainer (exact, not approximate)
- **Quality rating:** 4/5 — mathematically exact Shapley values for tree 
  ensembles; no approximation error
- **SHAP faithfulness score:** 55% (n=20 sample; directional evidence only)
- **SHAP stability score:** 0.0000 mean diff under small perturbations — 
  highly stable explanations
- **Top SHAP features:** StockOptionLevel, MaritalStatus_Married, 
  Department_Research & Development
- **Regulatory suitability:** MEDIUM — post-hoc explanation tool required; 
  not interpretable by design, but SHAP is mathematically exact

## Ethical Considerations
Same as Logistic Regression card. MaritalStatus appearing as a top SHAP 
feature warrants particular scrutiny in jurisdictions where marital status 
is a protected characteristic.

## Caveats and Recommendations
- Hyperparameter tuning (relaxing max_depth) recommended before concluding 
  Random Forest cannot outperform Logistic Regression on this task
- SHAP explanations are exact for this model type — suitable for audit use
- Same monitoring and retraining triggers as Logistic Regression