import pytest
import scanpy as sc
import anndata
import os

def test_data_loading():
    """Test data loading functionality"""
    # Test mouse data loading
    test_path = os.path.join("tests", "data", "test_mouse_data.h5ad")
    adata = sc.read_h5ad(test_path)
    
    # Basic assertions
    assert adata.n_obs > 0, "No cells loaded"
    assert adata.n_vars > 0, "No genes loaded"
    assert 'dataset' in adata.obs.columns, "Missing dataset metadata"

def test_preprocessing():
    """Test preprocessing pipeline"""
    test_path = os.path.join("tests", "data", "test_mouse_data.h5ad")
    adata = sc.read_h5ad(test_path)
    
    # Run preprocessing
    original_size = adata.shape
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Check filtering
    assert adata.shape[0] <= original_size[0], "Cell filtering failed"
    assert adata.shape[1] <= original_size[1], "Gene filtering failed"
    
    # Check normalization
    assert 'log1p' in adata.uns, "Log transformation not performed"

def test_clustering():
    """Test clustering functionality"""
    test_path = os.path.join("tests", "data", "test_mouse_data.h5ad")
    adata = sc.read_h5ad(test_path)
    
    # Run clustering
    n_clusters = 5
    from script import manual_kmeans_clustering  # Import your actual function
    manual_kmeans_clustering(adata, n_clusters=n_clusters)
    
    assert 'kmeans' in adata.obs.columns, "Clustering failed"
    assert len(adata.obs['kmeans'].cat.categories) == n_clusters, "Cluster count mismatch"

def test_annotation():
    """Test cell type annotation"""
    test_path = os.path.join("tests", "data", "test_mouse_data.h5ad")
    adata = sc.read_h5ad(test_path)
    
    # Run annotation
    from script import annotate_clusters, MOUSE_CELL_MARKERS
    annotate_clusters(adata, MOUSE_CELL_MARKERS)
    
    assert 'cell_type' in adata.obs.columns, "Annotation failed"
    assert adata.obs['cell_type'].nunique() > 1, "All cells same type"

def test_prediction():
    """Test prediction pipeline"""
    test_path = os.path.join("tests", "data", "test_mouse_data.h5ad")
    adata = sc.read_h5ad(test_path)
    
    # Run prediction
    from script import predict_cell_types
    model_path = "models/mouse/adipocyte_annotator_mouse_model.joblib"
    
    if os.path.exists(model_path):
        result = predict_cell_types(adata, model_path)
        assert 'predicted_cell_type' in result.obs.columns, "Prediction failed"
        assert 'prediction_confidence' in result.obs.columns, "Confidence missing"
    else:
        pytest.skip("Model not found for prediction test")