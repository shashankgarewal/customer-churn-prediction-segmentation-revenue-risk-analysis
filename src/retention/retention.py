from src.utils.config import ACTION_PLAYBOOK


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