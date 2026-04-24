

## Business Problem:
A mid-sized e-commerce business raised a concern: customers are leaving, and the team has no early warning system to identify who is at risk before it's too late.

Investigation revealed a serious cricise that Premium segment customers, who contribute the most revenue, experience the highest attrition at ~38%, while mid-tier customers remain the most retained. This creates a compounding revenue risk: not only are customers leaving, but the ones leaving are the most valuable.

The business needs a proactive way to identify at-risk customers and estimate their value — to ensure retention efforts are directed where they matter most.

> Note: This business context is fictional to showcase a holistic and real use of the project.

## Mitigation Approach:
A classification model will be developed to flag customers at risk of leaving, paired with a regression model to estimate their lifetime value — enabling the retention team to prioritize outreach toward High and Premium segment customers before they are lost.

Churn model prioritizes **Recall** (primary) and **ROC-AUC** (secondary), ensuring high-value customers at risk are not missed over overall accuracy. LTV model is evaluated on **MAE** for interpretable dollar-level error and **R²** for explained variance.