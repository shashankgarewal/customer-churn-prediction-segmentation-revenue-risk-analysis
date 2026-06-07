ACTION_PLAYBOOK = {

    "Dormant Customer": {

        "VIP": [
            "VIP Win-back Campaign",
            "Personalized Product Recommendations",
            "Priority Support Outreach"
        ],

        "IMP": [
            "Win-back Campaign",
            "Personalized Product Recommendations"
        ],

        "High": [
            "Win-back Campaign",
            "Personalized Product Recommendations"
        ],

        "Medium": [
            "Automated Re-engagement Email"
        ],

        "Low": [
            "Newsletter Reminder"
        ]
    },

    "Disengaged Customer": {

        "VIP": [
            "VIP Exclusive Collection",
            "Personalized Recommendations",
            "Mobile Push Campaign"
        ],

        "IMP": [
            "Personalized Recommendations",
            "Mobile Push Campaign"
        ],

        "High": [
            "Product Discovery Campaign"
        ],

        "Medium": [
            "Re-engagement Email"
        ],

        "Low": [
            "Newsletter"
        ]
    },

    "Frustrated Customer": {

        "VIP": [
            "Dedicated Customer Success Outreach",
            "Priority Support",
            "Returns Experience Review"
        ],

        "IMP": [
            "Priority Support",
            "Service Recovery Campaign"
        ],

        "High": [
            "Service Recovery Campaign"
        ],

        "Medium": [
            "Customer Feedback Survey"
        ],

        "Low": [
            "Feedback Request"
        ]
    },

    "Price Sensitive Shopper": {

        "VIP": [
            "VIP Loyalty Rewards",
            "Exclusive Discount Access"
        ],

        "IMP": [
            "Loyalty Rewards",
            "Targeted Coupon"
        ],

        "High": [
            "Targeted Coupon"
        ],

        "Medium": [
            "Promotional Campaign"
        ],

        "Low": [
            "Sale Notification"
        ]
    }
}

# Action mechanism to retain customer.
def _get_retention_actions(persona: str, segment: str) -> list[str]:

    return (
        ACTION_PLAYBOOK\
            .get(persona, {})\
                .get(segment, ["General Retention Campaign"])
    )                            

def recommend_action(segment: str, primary_persona: str) -> dict:

    actions = _get_retention_actions(
        persona=primary_persona,
        segment=segment
    )

    priority_map = {
        "VIP": "Critical",
        "IMP": "High",
        "High": "Medium",
        "Medium": "Low",
        "Low": "Minimal"
    }

    return {
        "persona": primary_persona,
        "priority": priority_map.get(segment, "Low"),
        "recommended_actions": actions
    }