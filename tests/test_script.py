import pytest
import scanpy as sc
import anndata
import os
import numpy as np
from adipocyte_annotator.script import (
    manual_kmeans_clustering,
    annotate_clusters,
    predict_cell_types,
    MOUSE_CELL_MARKERS,
    HUMAN_CELL_MARKERS
)

# Get list of all test files
TEST_DATA_DIR = os.path.join("tests", "data")
TEST_FILES = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith(".h5ad")]

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_data_loading(test_file):
    """Test data loading functionality"""
    test_path = os.path.join(TEST_DATA_DIR, test_file)
    adata = sc.read_h5ad(test_path)
    
    # Basic assertions
    assert adata.n_obs > 0, f"No cells loaded in {test_file}"
    assert adata.n_vars > 0, f"No genes loaded in {test_file}"
    assert 'cell_type' in adata.obs.columns, f"Missing cell type metadata in {test_file}"

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_preprocessing(test_file):
    """Test preprocessing pipeline"""
    test_path = os.path.join(TEST_DATA_DIR, test_file)
    adata = sc.read_h5ad(test_path)
    
    # Run preprocessing
    original_size = adata.shape
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Check filtering
    assert adata.shape[0] <= original_size[0], f"Cell filtering failed in {test_file}"
    assert adata.shape[1] <= original_size[1], f"Gene filtering failed in {test_file}"

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_clustering(test_file):
    """Test clustering functionality"""
    test_path = os.path.join(TEST_DATA_DIR, test_file)
    adata = sc.read_h5ad(test_path)
    
    # Run clustering
    n_clusters = 5
    manual_kmeans_clustering(adata, n_clusters=n_clusters)
    
    assert 'kmeans' in adata.obs.columns, f"Clustering failed in {test_file}"
    assert len(adata.obs['kmeans'].cat.categories) == n_clusters, f"Cluster count mismatch in {test_file}"

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_annotation(test_file):
    """Test cell type annotation"""
    test_path = os.path.join(TEST_DATA_DIR, test_file)
    adata = sc.read_h5ad(test_path)
    
    # Run annotation
    
    # Determine species from filename
    if "EWAT" in test_file or "iWAT" in test_file:
        annotate_clusters(adata, MOUSE_CELL_MARKERS)
    else:
        annotate_clusters(adata, HUMAN_CELL_MARKERS)
    
    assert 'cell_type' in adata.obs.columns, f"Annotation failed in {test_file}"
    assert adata.obs['cell_type'].nunique() > 1, f"All cells same type in {test_file}"

@pytest.mark.parametrize("test_file", TEST_FILES)
def test_prediction(test_file):
    """Test prediction pipeline"""
    test_path = os.path.join(TEST_DATA_DIR, test_file)
    adata = sc.read_h5ad(test_path)
    
    # Run prediction
    
    # Determine species from filename
    if "EWAT" in test_file or "iWAT" in test_file:
        model_path = "/workspaces/CS506_project/models/mouse/adipocyte_annotator_mouse_model.joblib"
        species = "mouse"
    else:
        model_path = "/workspaces/CS506_project/models/human/adipocyte_annotator_human_model.joblib"
        species = "human"
    
    if os.path.exists(model_path):
        result = predict_cell_types(adata, model_path, species=species)
        assert 'predicted_cell_type' in result.obs.columns, f"Prediction failed in {test_file}"
        assert 'prediction_confidence' in result.obs.columns, f"Confidence missing in {test_file}"
    else:
        pytest.skip(f"Model not found for {test_file}")