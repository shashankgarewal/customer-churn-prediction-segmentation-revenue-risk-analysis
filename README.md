

## Business Problem:
A mid-sized e-commerce business raised a concern: customers are leaving, and the team has no early warning system to identify who is at risk before it's too late.

Investigation revealed a serious cricise that Premium segment customers, who contribute the most revenue, experience the highest attrition at ~38%, while mid-tier customers remain the most retained. This creates a compounding revenue risk: not only are customers leaving, but the ones leaving are the most valuable.

The business needs a proactive way to identify at-risk customers and estimate their value — to ensure retention efforts are directed where they matter most.

Apart from efforts in retaining customer, business needs to identify why customer is leaving.

> Note: This business context is fictional to showcase a holistic and real use of the project.

## Mitigation Approach:
A classification model will be developed to flag customers at risk of leaving, paired with a regression model to estimate their lifetime value — enabling the retention team to prioritize outreach toward High and Premium segment customers before they are lost.

Churn model prioritizes **Recall** (primary), and **Average Precision Score** and **ROC-AUC** (secondary - assess overfit), ensuring high-value customers at risk are not missed over overall accuracy. LTV model is evaluated on **MAE** for interpretable dollar-level error and **R²** for explained variance.

* Initially opted for recall as optimization, which fix a threshold. As the project intent to use different thresholds for ltv segment, logloss is used as it not dependent on threshold while penalize predicted probability for how far it is from actual value. 

**Business Success Metric:** Total Lifetime Value of correctly identified at-risk High/Premium customers — quantifying revenue protected through early intervention.

## Assumption:
* `Churned` column represent whether the customer will churn or not in future time-window (e.g, next 1 year).
* `Lifetime_Value` column represent the forecast of value customer can generate in his forcible future or time-window.

## Challenges & Learning

* **Dropping invalid feature values** — directly filtering on valid conditions (e.g. `df[df['Age'] <= 100]`) silently drops missing values too, since NaN comparisons evaluate to False. 

  **Solution:** use explicit null-preserving conditions or handle missingness separately before applying filters.

* **Log transformation assumptions** — log transform is effective for continuous right-skewed tails, but not for disjoint outlier clusters. Applying it to a bimodal/disjoint distribution doesn't fix the shape — it just rescales both clusters. 

  **Learning:** Always visualise the distribution (boxplot) before deciding on a transform strategy. 
    
  **Solution:** RobustScaler is sufficient when the outliers are sparse or disjoint   and a flag already isolates the cluster as a model-usable signal.

## Tech Stack:
* great expectations: schema check and data validation
* mlflow: model experiment tracking, and model registry for production
* optuna: parametric hyperparameter adjustment for model fine tuning

## Model experiment and observation:
* In inital experiment for model family selection, ensembling based models - boosting family (xgboost) has performed best followed by bagging (random forest) which is marginally behind. Rest of the model, like decision tree or knn or logistic regression.
* In churn training pipeline, following algorithm are experimented:
  1. Random Forest and XGBoost - base ensemble model that performed well in model family selection.
  2. Histogram Gradient Boosting - As boosting ensemble perfomed best, historgram is also ...
  3. CatBoost - uses native categorical encoding, testing this can result an improvement
  4. LightGBM - 

## Note: 
The inference pipeline starts with validated and preprocessed data which is common for full dataset in hand without affecting  

## Future Steps:
* Currently the churn model consider each customer for equal importance, a weight based modeling for different segment/ltv/ratio can be used for retention focused  churn modeling.
* Alternate, instead implement a regression model on expected loss (ltv * churn).
* Although, clv segmentation based thresholds are emposed in the implementation, this causes a problem of seperation of continous variable with little to no gaps between values (e.g., 800 units). An approach that incorrporate clv aware retention strategy without segmentation would be better respesentation.
* Using config based experiment trails instead of parameters and value in the python script.
* Currently the threshold for ltv segment for prediction from probability are picked based on intuition and randomly, in future a distribution plot of model generated probability for true churn feature can be used to determine the threshold.

> Fix your business number when its affected by leaving customer. 