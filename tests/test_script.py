import sys
import os
import pytest
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adipocyte_annotator.script import predict_cell_types, MOUSE_CELL_MARKERS, HUMAN_CELL_MARKERS

TEST_DATA_DIR = os.path.join("tests", "data")
TEST_FILES = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith(".h5ad")]
TEST_RESULTS_DIR = os.path.join("tests", "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def evaluate_model_performance(y_true, y_pred, species, test_file):
    """Generate performance metrics and visualizations"""
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(TEST_RESULTS_DIR, f"{species}_report_{test_file}.csv"))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {test_file}")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(TEST_RESULTS_DIR, f"{species}_confusion_matrix_{test_file}.png"), bbox_inches='tight')
    plt.close()
    
    return report['accuracy']

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_model_performance(test_file):
    """Test model performance on validation datasets"""
    test_path = os.path.join(TEST_DATA_DIR, test_file)
    adata = sc.read_h5ad(test_path)
    
    # Determine species and model path
    if "EWAT" in test_file or "iWAT" in test_file:
        species = "mouse"
        model_path = "models/mouse/adipocyte_annotator_mouse_model.joblib"
        markers = MOUSE_CELL_MARKERS
    else:
        species = "human"
        model_path = "models/human/adipocyte_annotator_human_model.joblib"
        markers = HUMAN_CELL_MARKERS
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found for {test_file}")

    # NEW: Run required preprocessing and clustering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    manual_kmeans_clustering(adata, n_clusters=15)  # Adjust cluster count as needed

    # Run prediction
    result = predict_cell_types(adata, model_path, markers, species=species)
    
    # Rest of your test assertions...
    
    # Get true and predicted labels
    y_true = adata.obs['cell_type'].astype(str)
    y_pred = result.obs['predicted_cell_type'].astype(str)
    
    # Calculate accuracy
    accuracy = evaluate_model_performance(y_true, y_pred, species, test_file)
    
    # Assert minimum performance threshold (adjust as needed)
    assert accuracy >= 0.75, f"Model accuracy ({accuracy:.2f}) below threshold for {test_file}"
    
    # Additional assertion: At least 90% of cells should have confidence > 0.5
    high_confidence_ratio = (result.obs['prediction_confidence'] > 0.5).mean()
    assert high_confidence_ratio >= 0.9, f"High confidence predictions insufficient in {test_file}"

def generate_summary_report():
    """Generate consolidated report after all tests complete"""
    reports = []
    for f in os.listdir(TEST_RESULTS_DIR):
        if f.endswith(".csv") and "report" in f:
            df = pd.read_csv(os.path.join(TEST_RESULTS_DIR, f), index_col=0)
            df['dataset'] = f.split('_')[-1].replace('.csv', '')
            reports.append(df)
    
    if reports:
        combined = pd.concat(reports)
        summary = combined.groupby('dataset')['precision', 'recall', 'f1-score', 'support'].mean()
        summary.to_csv(os.path.join(TEST_RESULTS_DIR, "performance_summary.csv"))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=combined.reset_index(), x='dataset', y='accuracy')
        plt.title("Model Accuracy Across Test Datasets")
        plt.ylim(0, 1.0)
        plt.savefig(os.path.join(TEST_RESULTS_DIR, "accuracy_summary.png"))
        plt.close()

# Hook to generate summary after pytest completes
def pytest_sessionfinish(session, exitstatus):
    generate_summary_report()