import os
import gzip
import numpy as np
import pandas as pd
import os
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.manifold import TSNE
import joblib
import logging
from scipy import io, sparse
import anndata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define biological states (cell type markers)
BIOLOGICAL_STATES = {
    'white_adipocyte': ['subcutaneous', 'visceral', 'epididymal', 'wAT'],
    'brown_adipocyte': ['interscapular', 'BAT', 'UCP1+'],
    'beige_adipocyte': ['inguinal', 'brite', 'recruited'],
    'stromal_cells': ['SVF', 'CD34+', 'progenitor', 'preadipocyte'],
    'immune_cells': ['CD45+', 'macrophage', 'crown-like']
}

# Cell marker information
CELL_MARKERS = {
    'white_adipocyte': ['Lep', 'Pparg', 'Adipoq', 'Retn', 'Lpl', 'Fabp4', 'Plin1'],
    'brown_adipocyte': ['Ucp1', 'Cidea', 'Prdm16', 'Ppargc1a', 'Dio2', 'Cox8b', 'Elovl3'],
    'beige_adipocyte': ['Tmem26', 'Tbx1', 'Cd137', 'Slc27a1', 'Pat2', 'Cited1', 'Ear2'],
    'stromal_cells': ['Pdgfra', 'Cd34', 'Ly6a', 'Cd29', 'Cd44', 'Cd140a', 'Sca1'],
    'immune_cells': ['Ptprc', 'Cd68', 'Adgre1', 'Cd11c', 'Cd11b', 'Il6', 'Tnf']
}

# Datasets to use for training - updated with your actual file structure
GEO_DATASETS = {
    'GSE272938': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE272938_extracted',
        'datasets': [
            {
                'name': 'eWAT_DFAT',
                'files': {
                    'barcodes': 'GSM8415340_eWAT_DFAT_barcodes.tsv.gz',
                    'features': 'GSM8415340_eWAT_DFAT_features.tsv.gz',
                    'matrix': 'GSM8415340_eWAT_DFAT_matrix.mtx.gz'
                },
                'tissue': 'eWAT',
                'cell_type': 'white_adipocyte'
            },
            {
                'name': 'iWAT_DFAT',
                'files': {
                    'barcodes': 'GSM8415341_iWAT_DFAT_barcodes.tsv.gz',
                    'features': 'GSM8415341_iWAT_DFAT_features.tsv.gz',
                    'matrix': 'GSM8415341_iWAT_DFAT_matrix.mtx.gz'
                },
                'tissue': 'iWAT',
                'cell_type': 'white_adipocyte'
            },
            {
                'name': 'eWAT_SVF',
                'files': {
                    'barcodes': 'GSM8415342_eWAT_SVF_barcodes.tsv.gz',
                    'features': 'GSM8415342_eWAT_SVF_features.tsv.gz',
                    'matrix': 'GSM8415342_eWAT_SVF_matrix.mtx.gz'
                },
                'tissue': 'eWAT',
                'cell_type': 'stromal_cells'
            },
            {
                'name': 'iWAT_SVF',
                'files': {
                    'barcodes': 'GSM8415343_iWAT_SVF_barcodes.tsv.gz',
                    'features': 'GSM8415343_iWAT_SVF_features.tsv.gz',
                    'matrix': 'GSM8415343_iWAT_SVF_matrix.mtx.gz'
                },
                'tissue': 'iWAT',
                'cell_type': 'stromal_cells'
            }
        ]
    },
    'GSE266326': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE266326_extracted',
        'datasets': [
            {
                'name': 'Chow-Flu-5',
                'files': {
                    'h5': 'GSM8244976_Chow-Flu-5.h5'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': 'Chow-Flu-6',
                'files': {
                    'h5': 'GSM8244977_Chow-Flu-6.h5'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': 'Chow-Saline-1',
                'files': {
                    'h5': 'GSM8244978_Chow-Saline-1.h5'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': 'EWAT-CD45pos',
                'files': {
                    'h5': 'GSM8244980_EWAT-CD45pos.h5'
                },
                'tissue': 'eWAT',
                'cell_type': 'immune_cells'
            },
            {
                'name': 'HINI-EWAT-CD45pos',
                'files': {
                    'h5': 'GSM8244985_HINI-EWAT-CD45pos.h5'
                },
                'tissue': 'eWAT',
                'cell_type': 'immune_cells'
            }
        ]
    }
}

def configure_scanpy():
    """Configure scanpy to avoid numba errors"""
    try:
        # Disable numba JIT completely
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        
        # Set scanpy settings to handle potential issues
        sc.settings.verbosity = 0  # Reduce verbosity
        sc.settings.n_jobs = 1     # Avoid parallelization issues
        
        # Make sure variable names are unique
        sc.settings.var_names_make_unique = True
        
        logger.info("Scanpy configuration successful")
    except Exception as e:
        logger.error(f"Error configuring scanpy: {e}")

def manual_highly_variable_genes(adata, n_top_genes=2000):
    """
    Manually identify highly variable genes without using numba
    """
    logger.info(f"Manually selecting {n_top_genes} highly variable genes")
    
    # Extract matrix
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Calculate mean and variance
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    # Calculate coefficient of variation
    cv = variances / (means + 1e-8)
    
    # Get indices of top genes by coefficient of variation
    top_indices = np.argsort(cv)[-n_top_genes:]
    
    # Create highly_variable flag
    highly_variable = np.zeros(adata.n_vars, dtype=bool)
    highly_variable[top_indices] = True
    
    # Add to var
    adata.var['highly_variable'] = highly_variable
    adata.var['means'] = means
    adata.var['variances'] = variances
    adata.var['cv'] = cv
    
    # Create uns entry for consistency
    adata.uns['hvg'] = {'flavor': 'manual_variance'}
    
    logger.info(f"Selected {sum(highly_variable)} highly variable genes")
    
    return adata

def manual_pca(adata, n_comps=50):
    """
    Perform PCA manually without using scanpy's implementation
    """
    logger.info(f"Running manual PCA with {n_comps} components")
    
    # Convert to dense if sparse
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Standardize data
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds
    
    # Run PCA
    pca = PCA(n_components=min(n_comps, min(X.shape)))
    X_pca = pca.fit_transform(X_std)
    
    # Store in AnnData
    adata.obsm['X_pca'] = X_pca
    adata.uns['pca'] = {
        'variance_ratio': pca.explained_variance_ratio_,
        'variance': pca.explained_variance_
    }
    
    logger.info(f"PCA complete: explained variance ratio sum = {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return adata

def manual_tsne(adata, n_components=2, perplexity=30):
    """
    Run t-SNE manually without relying on scanpy
    """
    logger.info(f"Running manual t-SNE")
    
    # Check if PCA is available
    if 'X_pca' not in adata.obsm:
        logger.error("PCA not found in adata.obsm")
        return adata
    
    # Get PCA space
    X_pca = adata.obsm['X_pca']
    
    # Run t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Store in AnnData
    adata.obsm['X_tsne'] = X_tsne
    
    logger.info(f"t-SNE complete")
    
    return adata

def manual_kmeans_clustering(adata, n_clusters=20, key_added='kmeans'):
    """
    Perform clustering using KMeans
    """
    logger.info(f"Running KMeans clustering with {n_clusters} clusters")
    
    # Use PCA coordinates
    if 'X_pca' not in adata.obsm:
        logger.error("PCA not found in adata.obsm")
        return adata
    
    # Get PCA space
    X_pca = adata.obsm['X_pca']
    
    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    # Store in AnnData
    adata.obs[key_added] = labels.astype(str)
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    
    logger.info(f"KMeans clustering complete: {n_clusters} clusters")
    
    return adata

def read_gzipped_file(file_path):
    """Read a gzipped file properly"""
    with gzip.open(file_path, 'rb') as f:
        content = f.read().decode('utf-8')
    return [line for line in content.split('\n') if line]

def process_10x_data(dataset_info, base_dir):
    """
    Process 10X Genomics format data (matrix, features, barcodes)
    """
    name = dataset_info['name']
    files = dataset_info['files']
    logger.info(f"Processing 10X dataset: {name}")
    
    # Construct file paths
    barcodes_path = os.path.join(base_dir, files['barcodes'])
    features_path = os.path.join(base_dir, files['features'])
    matrix_path = os.path.join(base_dir, files['matrix'])
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [barcodes_path, features_path, matrix_path]):
        missing = [p for p in [barcodes_path, features_path, matrix_path] if not os.path.exists(p)]
        logger.error(f"Missing files for {name}: {missing}")
        return None
    
    try:
        logger.info(f"Reading files with gzip handling for {name}")
        
        # Read barcodes file
        logger.info(f"Reading barcodes file: {barcodes_path}")
        barcodes = read_gzipped_file(barcodes_path)
        
        # Read features file
        logger.info(f"Reading features file: {features_path}")
        features_lines = read_gzipped_file(features_path)
        features = []
        
        # Parse feature IDs and names
        for line in features_lines:
            parts = line.split('\t')
            if len(parts) > 1:
                # Most 10x files have gene IDs in first column, names in second
                features.append(parts[1])
            else:
                features.append(parts[0])
        
        # Read mtx file - use scipy.io
        logger.info(f"Reading matrix file: {matrix_path}")
        with gzip.open(matrix_path, 'rb') as f:
            X = io.mmread(f).T.tocsr()
        
        # Create AnnData object
        logger.info(f"Creating AnnData object with shape {X.shape}")
        adata = anndata.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=features))
        
        # Add dataset metadata
        adata.obs['dataset'] = name
        adata.obs['tissue'] = dataset_info['tissue']
        adata.obs['initial_type'] = dataset_info['cell_type']
        
        logger.info(f"Successfully loaded {name}: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
    
    except Exception as e:
        logger.error(f"Error loading 10X data for {name}: {e}")
        return None

def process_h5_data(h5_path, dataset_info):
    """
    Process H5 format data (10X Genomics H5 files)
    """
    name = dataset_info['name']
    logger.info(f"Processing H5 dataset: {name}")
    
    # Check if file exists
    if not os.path.exists(h5_path):
        logger.error(f"Missing H5 file for {name}: {h5_path}")
        return None
    
    try:
        # Check file size
        file_size = os.path.getsize(h5_path)
        if file_size == 0:
            logger.error(f"H5 file for {name} is empty (0 bytes)")
            return None
        
        # Try loading as 10X H5
        adata = sc.read_10x_h5(h5_path)
        logger.info(f"Successfully loaded {name} as 10X H5: {adata.shape[0]} cells, {adata.shape[1]} genes")
    except Exception as e:
        logger.warning(f"Failed to load as 10X H5, trying alternative format: {e}")
        try:
            # Try as AnnData H5AD
            adata = sc.read_h5ad(h5_path)
            logger.info(f"Successfully loaded {name} as H5AD: {adata.shape[0]} cells, {adata.shape[1]} genes")
        except Exception as e2:
            logger.error(f"Error loading H5 data for {name}: {e2}")
            return None
    
    # Add dataset metadata
    adata.obs['dataset'] = name
    adata.obs['tissue'] = dataset_info['tissue']
    adata.obs['initial_type'] = dataset_info['cell_type']
    
    return adata

def process_geo_data(geo_accession, data_dir='geo_data', max_cells_per_dataset=5000):
    """
    Process GEO dataset with multiple samples
    """
    logger.info(f"Processing {geo_accession} data")
    
    if geo_accession not in GEO_DATASETS:
        logger.error(f"No dataset info found for {geo_accession}")
        return None
    
    dataset_info = GEO_DATASETS[geo_accession]
    extracted_dir = os.path.join(data_dir, dataset_info['extracted_dir'])
    
    if not os.path.exists(extracted_dir):
        logger.error(f"Extracted directory {extracted_dir} not found")
        return None
    
    # Process each dataset in the GEO accession
    adatas = []
    
    for ds in dataset_info['datasets']:
        # Process based on file type
        if 'h5' in ds.get('files', {}):
            h5_path = os.path.join(extracted_dir, ds['files']['h5'])
            adata = process_h5_data(h5_path, ds)
        else:
            adata = process_10x_data(ds, extracted_dir)
        
        if adata is not None:
            # Basic preprocessing
            logger.info(f"Running basic preprocessing on {ds['name']}")
            
            # Make variable names unique
            adata.var_names_make_unique()
            
            # Filter cells and genes
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            
            # Normalize
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Subsample the dataset to make it manageable
            if adata.n_obs > max_cells_per_dataset:
                logger.info(f"Subsampling {ds['name']} from {adata.n_obs} to {max_cells_per_dataset} cells")
                adata = adata[np.random.choice(adata.n_obs, max_cells_per_dataset, replace=False)].copy()
            
            # Store raw data
            adata.raw = adata
            
            adatas.append(adata)
    
    if not adatas:
        logger.error(f"No valid data found for {geo_accession}")
        return None
    
    # If multiple adatas, concatenate them
    if len(adatas) > 1:
        try:
            # Add unique obs names to avoid conflicts
            for i, adata in enumerate(adatas):
                adata.obs_names_make_unique(f"dataset{i}_")
            
            combined_adata = sc.concat(adatas, join='outer')
            logger.info(f"Combined {len(adatas)} datasets for {geo_accession}: {combined_adata.shape[0]} cells, {combined_adata.shape[1]} genes")
            return combined_adata
        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            # Fall back to returning the first dataset
            logger.info(f"Falling back to using only the first dataset")
            return adatas[0]
    else:
        return adatas[0]

def preprocess_and_cluster(adata, n_clusters=20):
    """
    Preprocessing and clustering for single-cell data (without numba)
    """
    logger.info(f"Running preprocessing and clustering on dataset with {adata.shape[0]} cells")
    
    # Make sure we have raw data stored
    if adata.raw is None:
        adata.raw = adata
    
    # Find highly variable genes manually
    manual_highly_variable_genes(adata, n_top_genes=2000)
    
    # Filter to highly variable genes
    adata = adata[:, adata.var.highly_variable]
    
    # Run PCA manually
    manual_pca(adata, n_comps=50)
    
    # Run t-SNE for visualization
    manual_tsne(adata)
    
    # Run K-means clustering
    manual_kmeans_clustering(adata, n_clusters=n_clusters, key_added='kmeans')
    
    # Create a copy of kmeans as 'leiden' for compatibility with existing code
    adata.obs['leiden'] = adata.obs['kmeans']
    
    return adata

def annotate_clusters(adata, marker_dict=CELL_MARKERS):
    """
    Annotate clusters based on marker genes
    """
    logger.info("Annotating clusters based on marker genes")
    
    # Get cluster info
    clusters = adata.obs['leiden'].cat.categories
    
    # For each cluster, calculate marker gene scores
    cluster_scores = {}
    
    for cluster in clusters:
        cluster_scores[cluster] = {}
        
        # Get cells in this cluster
        cluster_cells = adata[adata.obs['leiden'] == cluster]
        
        # For each cell type, calculate marker score
        for cell_type, markers in marker_dict.items():
            # Find which markers are in our dataset
            markers_in_data = [m for m in markers if m in adata.var_names]
            
            if not markers_in_data:
                cluster_scores[cluster][cell_type] = 0
                continue
            
            # Calculate mean expression of marker genes in this cluster
            marker_mean = np.zeros(len(markers_in_data))
            
            for i, marker in enumerate(markers_in_data):
                # Handle both sparse and dense matrices
                if hasattr(cluster_cells.X, 'toarray'):
                    expr = cluster_cells[:, marker].X.toarray().flatten()
                else:
                    expr = cluster_cells[:, marker].X.flatten()
                marker_mean[i] = np.mean(expr)
            
            # Store average marker expression
            cluster_scores[cluster][cell_type] = np.mean(marker_mean)
    
    # Assign cell types based on highest marker score
    cluster_types = {}
    
    for cluster, scores in cluster_scores.items():
        if not scores:  # Empty scores
            cluster_types[cluster] = 'unknown'
            continue
            
        # Find cell type with highest score
        max_type = max(scores.items(), key=lambda x: x[1])
        
        # Only assign if score is above threshold
        if max_type[1] > 0.1:  # Adjust threshold as needed
            cluster_types[cluster] = max_type[0]
        else:
            cluster_types[cluster] = 'unknown'
    
    # Add cell type annotations to AnnData
    adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_types)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    # Compute confidence scores
    confidence_scores = {}
    
    for cluster, scores in cluster_scores.items():
        if not scores:
            confidence_scores[cluster] = 0
            continue
            
        max_score = max(scores.values())
        confidence_scores[cluster] = max_score
    
    adata.obs['confidence'] = adata.obs['leiden'].map(confidence_scores)
    
    # Show distribution of cell types
    cell_type_counts = adata.obs['cell_type'].value_counts()
    logger.info("Cell type distribution:")
    for cell_type, count in cell_type_counts.items():
        logger.info(f"  {cell_type}: {count} cells ({count/adata.shape[0]*100:.1f}%)")
    
    return adata

def prepare_features_for_training(adata):
    """
    Prepare features for training the random forest model from annotated data
    """
    logger.info("Preparing features for training")
    
    # Extract features for each cell
    features = []
    labels = []
    
    # For each cell, extract features
    for cell_idx in range(adata.shape[0]):
        # Get cell expression profile - handle both sparse and dense matrices
        if hasattr(adata.X, 'toarray'):
            cell_expr = adata.X[cell_idx].toarray()[0]
        else:
            cell_expr = adata.X[cell_idx]
        
        # Create feature dictionary
        feature_dict = {}
        
        # Get mean expression of marker genes for each cell type
        for ct, markers in CELL_MARKERS.items():
            # Find markers that are in our dataset
            markers_in_data = [m for m in markers if m in adata.var_names]
            
            if markers_in_data:
                # Get expression values for the markers
                marker_expr = []
                for marker in markers_in_data:
                    try:
                        marker_idx = adata.var_names.get_loc(marker)
                        marker_expr.append(cell_expr[marker_idx])
                    except:
                        # Skip if marker can't be found
                        continue
                
                if marker_expr:
                    # Calculate features
                    feature_dict[f"{ct}_mean_expr"] = np.mean(marker_expr)
                    feature_dict[f"{ct}_max_expr"] = np.max(marker_expr)
                    feature_dict[f"{ct}_expr_ratio"] = np.sum(np.array(marker_expr) > 0) / len(markers_in_data)
                else:
                    feature_dict[f"{ct}_mean_expr"] = 0
                    feature_dict[f"{ct}_max_expr"] = 0
                    feature_dict[f"{ct}_expr_ratio"] = 0
            else:
                feature_dict[f"{ct}_mean_expr"] = 0
                feature_dict[f"{ct}_max_expr"] = 0
                feature_dict[f"{ct}_expr_ratio"] = 0
        
        # Add some general features
        if 'n_genes' in adata.obs:
            feature_dict['n_genes'] = adata.obs['n_genes'][cell_idx]
        
        if 'n_counts' in adata.obs:
            feature_dict['total_counts'] = adata.obs['n_counts'][cell_idx]
        elif 'total_counts' in adata.obs:
            feature_dict['total_counts'] = adata.obs['total_counts'][cell_idx]
        else:
            feature_dict['total_counts'] = np.sum(cell_expr)
        
        # Get label (true cell type)
        cell_type = adata.obs['cell_type'][cell_idx]
        
        # Add to lists
        features.append(feature_dict)
        labels.append(cell_type)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    
    logger.info(f"Prepared {len(feature_df)} cells with {len(feature_df.columns)} features")
    
    return feature_df, np.array(labels)

def train_random_forest_model(features, labels, n_estimators=100, max_depth=None):
    """
    Train a random forest classifier
    """
    logger.info("Training random forest model")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Model training accuracy: {train_score:.4f}")
    logger.info(f"Model test accuracy: {test_score:.4f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification report:\n{report}")
    
    # Feature importance
    feature_importances = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Top 10 important features:\n{feature_importances.head(10)}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, features, labels, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f}")
    
    return model, (X_test, y_test, y_pred)

def save_model_and_artifacts(model, feature_names, output_dir='models'):
    """
    Save the trained model and related artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'adipocyte_annotator_model.joblib')
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    # Save cell markers
    marker_path = os.path.join(output_dir, 'cell_markers.json')
    with open(marker_path, 'w') as f:
        import json
        json.dump(CELL_MARKERS, f, indent=2)
    
    logger.info(f"Model and artifacts saved to {output_dir}")

def visualize_results(adata, output_dir='.'):
    """
    Create visualization plots
    """
    logger.info("Creating visualization plots")
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Check if UMAP was computed
    if 'X_umap' not in adata.obsm:
        logger.warning("UMAP not computed, skipping UMAP plots")
        return
    
    # UMAP colored by cell type
    plt.figure(figsize=(10, 8))
    sc.pl.umap(adata, color='cell_type', s=50, legend_loc='on data',
               palette='Set2', title='Cell Types', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'umap_cell_types.png'))
    plt.close()
    
    # UMAP colored by cluster
    plt.figure(figsize=(10, 8))
    sc.pl.umap(adata, color='leiden', s=50, legend_loc='on data',
               palette='tab20', title='Clusters', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'umap_clusters.png'))
    plt.close()
    
    # Select a few marker genes to visualize
    selected_markers = {}
    for ct, markers in CELL_MARKERS.items():
        # Take up to 2 markers from each cell type that are in the dataset
        markers_in_data = [m for m in markers if m in adata.var_names][:2]
        if markers_in_data:
            selected_markers[ct] = markers_in_data
    
    # Flatten the list of markers
    all_selected_markers = []
    for markers in selected_markers.values():
        all_selected_markers.extend(markers)
    
    # Create gene expression plots
    if all_selected_markers:
        # UMAP plots colored by gene expression
        for i, marker in enumerate(all_selected_markers):
            plt.figure(figsize=(8, 6))
            sc.pl.umap(adata, color=marker, s=50, color_map='viridis',
                      title=f'{marker} Expression', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'figures', f'umap_{marker}_expression.png'))
            plt.close()
        
        # Violin plots showing marker expression by cell type
        plt.figure(figsize=(14, 10))
        
        # Plot up to 10 markers to keep the plot readable
        plot_markers = all_selected_markers[:min(len(all_selected_markers), 10)]
        
        sc.pl.violin(adata, plot_markers, groupby='cell_type', 
                    rotation=90, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'violin_marker_genes.png'))
        plt.close()
    
    logger.info(f"Visualization plots saved to {output_dir}/figures/")

def main():
    """
    Main function to run the training pipeline
    """
    # Configure scanpy
    configure_scanpy()
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    
    # Path to data directory - UPDATE THIS to match your local path
    data_dir = 'geo_data'  # Default, change this if your files are in a different location
    
    # Process datasets
    adatas = []
    
    for geo_id in GEO_DATASETS:
        # Process dataset
        adata = process_geo_data(geo_id, data_dir)
        if adata is not None:
            # Run preprocessing and clustering
            adata = preprocess_and_cluster(adata)
            
            # Annotate clusters
            adata = annotate_clusters(adata)
            
            # Visualize
            visualize_results(adata)
            
            adatas.append(adata)
    
    if not adatas:
        logger.error("No datasets were successfully processed")
        return
    
    # Combine datasets if we have multiple
    if len(adatas) > 1:
        combined_adata = sc.concat(adatas, join='outer')
        logger.info(f"Combined all datasets: {combined_adata.shape[0]} cells, {combined_adata.shape[1]} genes")
    else:
        combined_adata = adatas[0]
    
    # Prepare features for training
    features, labels = prepare_features_for_training(combined_adata)
    
    # Train model
    model, eval_results = train_random_forest_model(features, labels)
    
    # Save model and artifacts
    save_model_and_artifacts(model, features.columns)
    
    # Create evaluation plots
    X_test, y_test, y_pred = eval_results
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    feature_importances = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    sns.barplot(data=feature_importances, x='importance', y='feature')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    logger.info("Training pipeline completed successfully")
    
    # Save the final annotated data
    combined_adata.write('annotated_data.h5ad')
    logger.info("Saved annotated data to annotated_data.h5ad")

if __name__ == "__main__":
    main()
