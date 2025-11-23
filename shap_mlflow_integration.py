"""
SHAP + MLflow Integration Module
Logs SHAP explanations as MLflow artifacts with proper visualization
"""

import mlflow
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json

class SHAPMLflowLogger:
    """
    Integrates SHAP explainability with MLflow tracking
    """
    
    def __init__(self, model, X_train, X_test, feature_names=None):
        """
        Initialize SHAP logger
        
        Args:
            model: Trained model (tree-based or any sklearn model)
            X_train: Training data for SHAP background
            X_test: Test data for SHAP analysis
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_test.shape[1])]
        self.shap_values = None
        self.explainer = None
        
    def compute_shap_values(self, model_type='tree'):
        """
        Compute SHAP values based on model type
        
        Args:
            model_type: 'tree', 'kernel', or 'linear'
        """
        print("Computing SHAP values...")
        
        if model_type == 'tree':
            # For tree-based models (RandomForest, XGBoost, etc.)
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.X_test)
        elif model_type == 'kernel':
            # For any model (slower but universal)
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            self.shap_values = self.explainer.shap_values(self.X_test)
        elif model_type == 'linear':
            # For linear models
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
            self.shap_values = self.explainer.shap_values(self.X_test)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        print(f"✓ SHAP values computed. Shape: {np.array(self.shap_values).shape}")
        return self.shap_values
    
    def log_summary_plot(self, plot_type='bar', max_display=10):
        """
        Create and log SHAP summary plot
        
        Args:
            plot_type: 'bar', 'dot', or 'violin'
            max_display: Maximum number of features to display
        """
        plt.figure(figsize=(10, 6))
        
        if isinstance(self.shap_values, list):
            # Multi-class case - use first class
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.summary_plot(
            shap_vals, 
            self.X_test, 
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), f"shap_summary_{plot_type}.png")
        plt.close()
        print(f"✓ Logged SHAP summary plot ({plot_type})")
    
    def log_feature_importance(self):
        """
        Calculate and log global feature importance from SHAP values
        """
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(self.shap_values[0]).mean(axis=0)
        else:
            shap_vals = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': shap_vals
        }).sort_values('importance', ascending=False)
        
        # Log as CSV
        importance_df.to_csv('shap_feature_importance.csv', index=False)
        mlflow.log_artifact('shap_feature_importance.csv', 'explainability')
        
        # Log top features as metrics
        for idx, row in importance_df.head(5).iterrows():
            mlflow.log_metric(f"shap_importance_{row['feature']}", row['importance'])
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'].head(10), importance_df['importance'].head(10))
        plt.xlabel('Mean |SHAP value|')
        plt.title('Top 10 Feature Importance (SHAP)')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), 'shap_feature_importance.png')
        plt.close()
        
        print(f"✓ Logged feature importance for {len(importance_df)} features")
        return importance_df
    
    def log_dependence_plots(self, top_n=3):
        """
        Create and log SHAP dependence plots for top N features
        
        Args:
            top_n: Number of top features to plot
        """
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Get top features by mean absolute SHAP value
        mean_shap = np.abs(shap_vals).mean(axis=0)
        top_features_idx = np.argsort(mean_shap)[-top_n:][::-1]
        
        for idx in top_features_idx:
            feature_name = self.feature_names[idx]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                idx,
                shap_vals,
                self.X_test,
                feature_names=self.feature_names,
                show=False
            )
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), f'shap_dependence_{feature_name}.png')
            plt.close()
        
        print(f"✓ Logged {top_n} dependence plots")
    
    def log_force_plots(self, sample_indices=None, num_samples=5):
        """
        Create and log SHAP force plots for individual predictions
        
        Args:
            sample_indices: Specific indices to plot (if None, random samples)
            num_samples: Number of samples to plot
        """
        if sample_indices is None:
            sample_indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
            expected_value = self.explainer.expected_value[0]
        else:
            shap_vals = self.shap_values
            expected_value = self.explainer.expected_value
        
        for i, idx in enumerate(sample_indices):
            plt.figure(figsize=(12, 3))
            shap.force_plot(
                expected_value,
                shap_vals[idx],
                self.X_test[idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), f'shap_force_plot_sample_{i}.png')
            plt.close()
        
        print(f"✓ Logged {len(sample_indices)} force plots")
    
    def log_waterfall_plot(self, sample_idx=0):
        """
        Create and log SHAP waterfall plot for a single prediction
        
        Args:
            sample_idx: Index of sample to explain
        """
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
            expected_value = self.explainer.expected_value[0]
        else:
            shap_vals = self.shap_values
            expected_value = self.explainer.expected_value
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[sample_idx],
                base_values=expected_value,
                data=self.X_test[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), f'shap_waterfall_sample_{sample_idx}.png')
        plt.close()
        
        print(f"✓ Logged waterfall plot")
    
    def log_all_plots(self, model_type='tree'):
        """
        Convenience method to compute SHAP and log all visualizations
        
        Args:
            model_type: Type of SHAP explainer to use
        """
        print("\n" + "="*60)
        print("SHAP Analysis & MLflow Logging")
        print("="*60)
        
        # Compute SHAP values
        self.compute_shap_values(model_type=model_type)
        
        # Log all visualizations
        self.log_summary_plot(plot_type='bar')
        self.log_summary_plot(plot_type='dot')
        self.log_feature_importance()
        self.log_dependence_plots(top_n=3)
        self.log_force_plots(num_samples=3)
        
        try:
            self.log_waterfall_plot(sample_idx=0)
        except:
            print("⚠ Waterfall plot not supported in this SHAP version")
        
        # Save raw SHAP values
        if isinstance(self.shap_values, list):
            for i, vals in enumerate(self.shap_values):
                np.save(f'shap_values_class_{i}.npy', vals)
                mlflow.log_artifact(f'shap_values_class_{i}.npy', 'explainability')
        else:
            np.save('shap_values.npy', self.shap_values)
            mlflow.log_artifact('shap_values.npy', 'explainability')
        
        print("\n✅ All SHAP artifacts logged to MLflow!")
        print("="*60 + "\n")


# Example usage function
def log_model_with_shap(model, X_train, X_test, y_train, y_test, 
                       feature_names, model_name='model', model_type='tree'):
    """
    Complete workflow: train model, evaluate, and log with SHAP
    
    Args:
        model: Trained sklearn model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        feature_names: List of feature names
        model_name: Name for the model
        model_type: Type for SHAP explainer
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    with mlflow.start_run(run_name=f"{model_name}_with_shap"):
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Predictions and metrics
        y_pred = model.predict(X_test)
        
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
        
        # SHAP Analysis
        shap_logger = SHAPMLflowLogger(model, X_train, X_test, feature_names)
        shap_logger.log_all_plots(model_type=model_type)
        
        print(f"✅ Run logged with ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("SHAP-MLflow Integration Module")
    print("Import this module and use SHAPMLflowLogger or log_model_with_shap()")
    print("\nExample:")
    print("  from shap_mlflow_integration import log_model_with_shap")
    print("  log_model_with_shap(model, X_train, X_test, y_train, y_test, feature_names)")
