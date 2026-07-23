import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class VisualizationHelper:
    @staticmethod
    def plot_clusters_2d(df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str, output_path: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette='viridis', style=cluster_col)
        plt.title(f"Cluster Visualization: {x_col} vs {y_col}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, output_path: str):
        plt.figure(figsize=(12, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_elbow_method(ks: list, inertias: list, output_path: str):
        plt.figure(figsize=(8, 5))
        plt.plot(ks, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method showing the optimal K')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
