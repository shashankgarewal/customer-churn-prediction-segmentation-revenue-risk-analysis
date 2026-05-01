

## Business Problem:
A mid-sized e-commerce business raised a concern: customers are leaving, and the team has no early warning system to identify who is at risk before it's too late.

Investigation revealed a serious cricise that Premium segment customers, who contribute the most revenue, experience the highest attrition at ~38%, while mid-tier customers remain the most retained. This creates a compounding revenue risk: not only are customers leaving, but the ones leaving are the most valuable.

The business needs a proactive way to identify at-risk customers and estimate their value — to ensure retention efforts are directed where they matter most.

Apart from efforts in retaining customer, business needs to identify why customer is leaving.

> Note: This business context is fictional to showcase a holistic and real use of the project.

## Mitigation Approach:
A classification model will be developed to flag customers at risk of leaving, paired with a regression model to estimate their lifetime value — enabling the retention team to prioritize outreach toward High and Premium segment customers before they are lost.

Churn model prioritizes **Recall** (primary) and **ROC-AUC** (secondary), ensuring high-value customers at risk are not missed over overall accuracy. LTV model is evaluated on **MAE** for interpretable dollar-level error and **R²** for explained variance.

**Business Success Metric:** Total Lifetime Value of correctly identified at-risk High/Premium customers — quantifying revenue protected through early intervention.

## Assumption:
* `Churned` column represent whether the customer will churn or not in future time-window (e.g, next 1 year).
* `Lifetime_Value` column represent the forecast of value customer can generate in his forcible future or time-window.

## Challenges & Learning
* Dropping invalid feature values - direct filter on valid cases drops missing values as well, as condition makes False for missing/nan values.
* 

## Challenges & Learning

* **Dropping invalid feature values** — directly filtering on valid conditions (e.g. `df[df['Age'] <= 100]`) silently drops missing values too, since NaN comparisons evaluate to False. 

  **Solution:** use explicit null-preserving conditions or handle missingness separately before applying filters.

* **Log transformation assumptions** — log transform is effective for continuous right-skewed tails, but not for disjoint outlier clusters. Applying it to a bimodal/disjoint distribution doesn't fix the shape — it just rescales both clusters. 

    **Learning:** Always visualise the distribution (boxplot) before deciding on a transform strategy. 
    
    **Solution:** RobustScaler is sufficient when the outliers are sparse or disjoint   and a flag already isolates the cluster as a model-usable signal.