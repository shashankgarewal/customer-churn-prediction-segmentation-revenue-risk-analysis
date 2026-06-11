# Customer Churn Prediction, Segmentation & Revenue Risk Analysis

An end-to-end machine learning system that predicts customer churn, identifies behavioral personas through SHAP-based driver analysis, recommends segment-specific retention strategies, and quantifies the financial exposure of predicted churn across LTV tiers.

Built to demonstrate how ML moves beyond accuracy metrics — from model output to business decision.

> From churn signal to retention action — before revenue walks out the door.

---

## Business Problem:
A mid-sized e-commerce business raised a concern: customers are leaving, and the team has no early warning system to identify who is at risk before it's too late.

Investigation revealed a serious crisis: Premium segment customers, who contribute the most revenue, experience the highest attrition at ~38%, while mid-tier customers remain the most retained. This creates a compounding revenue risk: not only are customers leaving, but the ones leaving are the most valuable.

The business needs a proactive way to identify at-risk customers and estimate their value — to ensure retention efforts are directed where they matter most.

Apart from efforts in retaining customer, business needs to identify why customer is leaving.

> **Note**: This business context is fictional to showcase a holistic and real-world use case of the project.

---

## Mitigation Approach

A classification model will be developed to flag customers at risk of leaving, paired with a regression model to estimate their lifetime value — enabling the retention team to prioritize outreach toward High and Premium segment customers before they are lost.

1. **Initial approach — Recall-optimized** (`recall-experiment-metric` git branch)  
   Optimized for Recall with **Average Precision Score** as primary evaluation metric and **ROC-AUC** as secondary (to assess overfitting). The rationale was to ensure high-value at-risk customers were not missed, even at the cost of additional false positives among lower-value customers.

2. **Revised approach — Segment-aware, threshold-decoupled**  
   As the project evolved toward LTV-segment-specific thresholds for churn classification, **Log Loss** became the primary metric. Unlike threshold-dependent metrics, Log Loss evaluates the quality of predicted churn probabilities directly, penalizing predictions proportionally to their confidence error.  
   
   **ROC-AUC** and **Average Precision Score per segment** are retained as secondary metrics for monitoring model discrimination and supporting production model selection.

### Why not use Recall, F1, or Global Metrics?

- **Recall / F1 Score**  
  These are single-threshold metrics (commonly evaluated at 0.5), which obscures model behavior across the full probability distribution. In production, optimal operating thresholds are rarely fixed at 0.5 and often vary by business segment and intervention cost.

- **Global ROC-AUC / Average Precision**  
  Aggregate metrics can hide poor performance within smaller or underrepresented customer segments. Larger groups dominate the overall score, masking segment-level failures (a form of Simpson’s Paradox at the metric level).  
   
  Additionally, **Average Precision** is prevalence-dependent, making comparisons across segments unreliable when churn rates differ substantially.

### Business Success Metric

- **Initial definition:**  
  Total LTV of correctly identified at-risk High/Premium customers, intended to estimate revenue preserved through early intervention.

- **Revised definition:**  
  Expected financial risk per churned customer calculated as `Risk = LTV × Churn Probability × Churn Prediction (0/1)`. This formulation reflects that customer value is not lost immediately upon churn but over a future time horizon. Summing raw LTV therefore overestimates short-term exposure. Weighting by predicted churn probability incorporates model confidence directly into the business impact metric.

---

## Assumption:
* `Churned` column represent whether the customer will churn or not in future time-window (e.g, next 1 year).
* `Lifetime_Value` column represent the forecast of value customer can generate in his forcible future or time-window.

---
## Challenges & Learning

* **Dropping invalid feature values** — directly filtering on valid conditions (e.g. `df[df['Age'] <= 100]`) silently drops missing values too, since NaN comparisons evaluate to False. 

  **Solution:** use explicit null-preserving conditions or handle missingness separately before applying filters.

* **Log transformation assumptions** — log transform is effective for continuous right-skewed tails, but not for disjoint outlier clusters. Applying it to a bimodal/disjoint distribution doesn't fix the shape — it just rescales both clusters. 

  **Learning:** Always visualise the distribution (boxplot) before deciding on a transform strategy. 
    
  **Solution:** RobustScaler is sufficient when the outliers are sparse or disjoint and a flag already isolates the cluster as a model-usable signal.

* **ROC-AUC and Average Precision Score — aggregate scores hide segment failures**  
  A single global AUC or Average Precision score is implicitly size-weighted across segments — larger segments dominate, masking poor model performance on smaller or underrepresented subgroups. Additionally, Average Precision 
  is prevalence-dependent, making cross-segment comparison unreliable when churn rates differ by LTV tier.

  **Learning:** Global AUC can look acceptable while the model is effectively failing on the segment that matters most (e.g. High/Premium customers). Aggregate metrics should never be the sole basis for model selection on segmented business problems.

  **Solution:** Compute ROC-AUC and Average Precision Score **per LTV segment** as secondary metrics alongside a threshold-independent primary metric (Log Loss) — exposing per-group model confidence without being tied to any single operating threshold.

* **The double-influence problem with LTV** — using Lifetime Value both as a sample weight and as a top input feature creates a feedback loop where the model's segment-level behavior is shaped twice by the same signal. The feature teaches the model what LTV means; the weight then amplifies its influence on the loss function.

  **Learning:** When a feature is also used to weight training examples, validate that the model is learning behavioral patterns — not just optimizing toward the weight signal itself.

* **SHAP for persona discovery, not explainability** — most churn projects use SHAP to answer "why did the model predict churn for this customer?" This project uses it differently: positive SHAP contributions identify which behavioral signals are actively driving churn risk, and those signals are matched against persona rules to classify the *type* of churn risk. The persona system — not the SHAP plot — is what drives the retention decision.

  **Learning:** SHAP's local attribution values are a rich behavioral signal in their own right, not just a model debugging tool.

---
  
## Tech Stack:
* **Great Expectations**: Schema enforcement, data profiling, and pipeline data validation.
* **MLFlow**: Model experiment tracking, artifact logging, and model registry for production deployment.
* **Optuna**: Parametric hyperparameter optimization and automated model fine-tuning.
* **SHAP**: Model explainability and retention strategy. 

---
## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ecom-customer-behaviour
   ```

2. **Initialize virtual environment:**
   ```bash
   # Create venv
   python -m venv venv

   # Activate venv
   # Windows:
   .\venv\Scripts\activate
   # Unix/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Preparation:**
   Ensure the raw dataset is placed in the `data/raw/` directory:
   `data/raw/ecommerce_customer_churn_dataset.csv`

---
## How to Run

All commands should be executed from the project root directory.

### 1. Training Pipeline
Runs the full workflow: data validation, preprocessing, feature engineering, and hyperparameter tuning. Results and models are tracked in MLflow.
```bash
python -m src.pipeline.churn_training --trials {n_trials} --models {support: randomforest xgboost catboost lightgbm hgboost}
```
*Parameters:*
- `--data-path`: (Optional) Path to raw data CSV relative from project root.
- `--models`: (Optional) Space-separated list of models to train (e.g., `xgboost catboost lightgbm`).
- `--trials`: (Optional) Number of Optuna optimization trials per model.

### 2. Inference Pipeline
Executes model prediction, business impact analysis, and SHAP-based retention strategy generation. The pipeline supports decoupled output modes to serve both high-level analytical summaries and granular operational records.

```bash
python -m src.pipeline.churn_inference
```

### 3. SHAP Explainability
Shows SHAP global mean values and distributions
```bash
python -m script.shap_global_importance
```

### 4. Reset Experiment (Optional)
Wipes all MLflow metadata and local artifacts for a clean start.
```bash
python -m script.reset_experiment --name churn-training-pipeline
```

---
## Model Experiments & Observations

### Family Selection (Initial Experiment)
Evaluated across major model families to narrow the search space before 
full pipeline experimentation. Ensemble methods dominated:

- **Boosting (XGBoost)** — best overall performance
- **Bagging (Random Forest)** — marginally behind XGBoost
- **KNN (distance based), Logistic Regression (boundary based)** — notably weaker even after compatible transformation, insufficient for the complexity of churn signal in this dataset
- **Decision Tree (split based)** - peformed better than distance and boundary based (non-ensemble) algorithm. 

### Churn Training Pipeline — Algorithms Evaluated

1. **Random Forest & XGBoost** — carried forward from family selection as strong baselines; set the performance floor for all subsequent experiments.

2. **Histogram Gradient Boosting (sklearn)** — bin-based boosting variant; faster training on larger data, native support for missing values without imputation. Evaluated to test if the speed-accuracy tradeoff favors it over XGBoost at this dataset scale.

3. **LightGBM** — leaf-wise tree growth (vs. depth-wise in XGBoost) makes it faster and often more accurate on large tabular datasets with lower memory footprint. Evaluated as a direct boosting alternative to XGBoost.

4. **CatBoost** — native ordered categorical encoding without preprocessing. Evaluated to test whether catboost native categorical preprocessing improves signal on LTV segment and contract-type features.

### Hyperparameter Tuning
Bayesian optimization via Optuna (TPE sampler) with stratified 3-fold cross-validation. Stratification on combined segment × churn label to preserve segment distribution across folds.

Two-phase approach: broad search (45 trials) followed by a focused run on tightened ranges derived from `slice plot analysis`.

### Final Model Selection
**CatBoost** selected as the final model based on overall AP and segment-level performance.

All gradient boosting variants converged to similar overall AP (~0.916), making segment-level metrics the deciding factor.

| Metric | CatBoost | XGBoost | LightGBM | HGBoost |
|---|---|---|---|---|
| AP | 0.9161 | 0.9155 | 0.9162 | 0.9149 |
| ROC-AUC | 0.9309 | 0.9295 | 0.9300 | 0.9292 |
| ap_imp | 0.9626 | 0.9629 | 0.9622 | 0.9611 |
| ap_vip | 0.9328 | 0.9077 | 0.9083 | 0.9272 |
| ap_high | 0.9709 | 0.9705 | 0.9708 | 0.9696 |
| ap_medium | 0.8111 | 0.8073 | 0.8074 | 0.8092 |
| log_loss | 0.2342 | 0.2360 | 0.2345 | 0.2371 |

CatBoost's advantage is most pronounced on the VIP segment (`ap_vip: 0.9328` vs XGBoost's `0.9077`) — the highest business-value customers, while being 

Log loss is slightly higher than CatBoost's best run, however the business action is a binary intervention, so ranking quality (AP/ROC) matters more than probability calibration.

### Segment Performance Note
Medium segment shows consistently weaker performance (AP ~0.81) across all models. 

This reflects the underlying data reality — medium customers have a 17.4% churn rate vs 35–40% for high/low/IMP segments. Lower churn rate means fewer consistent positive examples, making the churn signal inherently noisier for this group. Not a modeling failure — a reflection of business behavior.

---
## Ethical Considerations

**Return Rate and Service Calls: Two Different Behavioral Signals**

Global SHAP analysis reveals an important distinction between two behavioral features that might appear similar on the surface.

**Higher** customer **service calls** correlate with **lower churn** probability. Customers who reach out to support are actively engaged with the platform and less likely to leave silently. Contact is a retention signal, not a churn signal.

**Higher return** rates correlate with **higher churn** probability. Unlike service calls, returns appear to reflect product dissatisfaction or fulfillment failures rather than engagement. Customers who return frequently are at slightly elevated churn risk.

> E-commerce platforms that penalize high-return customers through return fees, account restrictions, or stringent claim reviews risk accelerating churn in an already at-risk group. A customer returning products due to platform-side failures (damaged goods, wrong items, fulfillment errors) is being penalized for the platform's mistake, which compounds the dissatisfaction already driving their churn risk.

Return rates provide useful signals, but they should be interpreted alongside **return reasons**.

---

## Future Steps and Approaches:
* **Metric-driven Retention Strategy**: Currently, the retention strategy is static and rule-based campaign assigment. Future versions, similar to metric driven persona, can implement metric driven rentention strategies by employing feature importance, retention signals, and/or partial dependency to identify best plausible feature/approach for customer intervention.
* **Campaign-Aware Retention Modeling:** Extend retention rate assumptions into a segment × persona × campaign matrix with per-campaign budget constraints and expected value optimization — moving from rule-based to ROI-maximizing intervention selection.
* **Balance Class each Segment:** The medium ltv customer segment have low proportion of churn and consistently ml algorithms are weak at capturing them. Undersampling the non-churn class sample of medium ltv segment would allow model to capture pattern better while using log-loss as optimization metric.
* **LTV-Weighted Modeling:** Currently, the model treats all customers as equally important. Future versions could use sample weights based on LTV so the model prioritizes high-value customers during training.
* **Expected Loss Regression:** Move from simple classification to a regression approach that predicts "Expected Financial Loss" (LTV × Churn Probability). This would allow the business to rank customers by actual dollar risk.
* **Continuous Retention Strategies:** Current LTV segments create "boundary problems" where a small difference in value (e.g., \$799 vs \$801) leads to a different strategy. We aim to move toward strategies that treat LTV as a continuous variable.
* **Config-Driven Experiments:** Move hyperparameter search spaces and Optuna settings into YAML configuration files to make experiments easier to manage without changing the Python code. Also, tracking and logging the search space for each optuna study. 
* **Automated Threshold Optimization:** Instead of picking probability thresholds based on intuition, use precision-recall curves and probability distributions to find the mathematically optimal threshold for each business segment.