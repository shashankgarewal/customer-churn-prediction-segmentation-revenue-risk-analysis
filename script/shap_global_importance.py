import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.pipeline.churn_inference import _get_model, _prepare_inference_data
from src.utils.common import get_project_root

if __name__ == "__main__":
    model, model_type = _get_model()
    X_test, _, _, _, _ = _prepare_inference_data(data=None, n_samples=np.inf, model=model, model_type=model_type)
    
    # Calculate SHAP values for global importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Compatibility with binary classification outputs (list of arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    global_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_dir_shap": shap_values.mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nGlobal SHAP Importance (Top 20):")
    print(global_importance.head(20))
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    
    save_path = get_project_root() / "docs" / "shap" / "summary_mean_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved SHAP Summary Mean Plot at {save_path}")

    try:
        print(
            "\nNative Model Feature Importance:\n",
            pd.Series(model.feature_importances_, model.feature_names_in_).sort_values(ascending=False).head(20)
        )
    except Exception:
        pass

    # SHAP Summary Plot (Beeswarm)
    print("\nGenerating SHAP Summary Plot (Beeswarm)... (Close the window to finish)")
    shap.summary_plot(shap_values, X_test, show=False)

    save_path = get_project_root() / "docs" / "shap" / "summary_dist_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()