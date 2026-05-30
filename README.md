

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
  
## Tech Stack:
* **Great Expectations**: Schema enforcement, data profiling, and pipeline data validation.
* **MLFlow**: Model experiment tracking, artifact logging, and model registry for production deployment.
* **Optuna**: Parametric hyperparameter optimization and automated model fine-tuning.

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
python -m src.pipeline.churn_training --trials 15 --models xgboost catboost
```
*Parameters:*
- `--data-path`: (Optional) Path to raw data CSV.
- `--models`: (Optional) Space-separated list of models to train (e.g., `xgboost catboost lightgbm`).
- `--trials`: (Optional) Number of Optuna optimization trials per model.

### 2. Inference Pipeline
Loads the 'best' model from the MLflow Registry (with local fallback) and evaluates it against the test set, including business impact analysis.
```bash
python -m src.pipeline.churn_inference
```

### 3. Reset Experiment
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

## Future Steps:
* **LTV-Weighted Modeling:** Currently, the model treats all customers as equally important. Future versions could use sample weights based on LTV so the model prioritizes high-value customers during training.
* **Expected Loss Regression:** Move from simple classification to a regression approach that predicts "Expected Financial Loss" (LTV × Churn Probability). This would allow the business to rank customers by actual dollar risk.
* **Continuous Retention Strategies:** Current LTV segments create "boundary problems" where a small difference in value (e.g., \$799 vs \$801) leads to a different strategy. We aim to move toward strategies that treat LTV as a continuous variable.
* **Config-Driven Experiments:** Move hyperparameter search spaces and Optuna settings into YAML configuration files to make experiments easier to manage without changing the Python code. Also, tracking and logging the search space for each optuna study. 
* **Automated Threshold Optimization:** Instead of picking probability thresholds based on intuition, use precision-recall curves and probability distributions to find the mathematically optimal threshold for each business segment.

> Fix your business number when its affected by leaving customer. 