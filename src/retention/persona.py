from typing import Dict

PERSONA_RULES = {
    "Dormant Customer": [
        "Days_Since_Last_Purchase",
        "Login_Frequency",
        "Credit_Balance"
    ],

    "Disengaged Customer": [
        "Login_Frequency",
        "Session_Duration_Avg",
        "Pages_Per_Session",
        "Email_Open_Rate",
        "Mobile_App_Usage",
        "Social_Media_Engagement_Score"
    ],

    "Price Sensitive Shopper": [
        "Discount_Usage_Rate",
        "Cart_Abandonment_Rate",
        "Wishlist_Items",
        "Average_Order_Value"
    ],

    "Frustrated Customer": [
        "Returns_Rate",
        "Customer_Service_Calls",
        "Product_Reviews_Written"
    ],
}


# determine persona of customer using SHAP contribution values

def _score_persona(features, drivers):
    """compute score and rank of customer persona"""
    matched_values = [drivers[f] for f in features if f in drivers]
    
    if not matched_values:
        return 0
    
    # coverage normalize for the feature-count bias between personas
    coverage = len(matched_values) / len(features)
    # proportional strength
    strength = sum(matched_values) / sum(drivers.values())
    
    return round(coverage * strength, 4)

def assign_persona(all_churn_drivers: Dict[str, float]) -> dict:
    
    persona_scores = {
        persona: _score_persona(features, all_churn_drivers) 
        for persona, features in PERSONA_RULES.items()
    }

    ranked = sorted(persona_scores.items(), key=lambda x: x[1], reverse=True)

    # Handle edge case where a customer has no churn driving feature mapping to any persona rules
    if not ranked or ranked[0][1] == 0:
        return {
            "primary_persona": "Unclassified / General Churn",
            "secondary_persona": None,
            "persona_scores": dict(ranked)
        }
        
    return {
        "primary_persona": ranked[0][0],
        "secondary_persona": ranked[1][0],
        "persona_scores": dict(ranked)
    }
    
