# Datasets to use for training MOUSE model
MOUSE_GEO_DATASETS = {
    'GSE272938': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE272938_extracted',  # This directory exists and has files
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
        'extracted_dir': 'GSE266326_extracted',  # This directory exists and has files
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
    },
    'GSE280109': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE280109_temp_extraction',  # Use the extracted directory
        'datasets': [
            {
                'name': 'adipose_innate_T_cells',
                'files': {
                    'barcodes': 'GSM8587430_barcodes.tsv.gz',
                    'features': 'GSM8587430_features.tsv.gz',
                    'matrix': 'GSM8587430_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'immune_cells'
            }
        ]
    },
    'GSE236579': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE236579_temp_extraction',
        'datasets': [
            {
                'name': 'CCC',
                'files': {
                    'barcodes': 'GSM7558353_CCC_barcodes.tsv.gz',
                    'features': 'GSM7558353_CCC_features.tsv.gz',
                    'matrix': 'GSM7558353_CCC_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': 'CC',
                'files': {
                    'barcodes': 'GSM7558355_CC_barcodes.tsv.gz',
                    'features': 'GSM7558355_CC_features.tsv.gz',
                    'matrix': 'GSM7558355_CC_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': 'HHC',
                'files': {
                    'barcodes': 'GSM7558356_HHC_barcodes.tsv.gz',
                    'features': 'GSM7558356_HHC_features.tsv.gz',
                    'matrix': 'GSM7558356_HHC_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            }
        ]
    },
    # New dataset: GSE214982 - BMPER marker
    'GSE214982': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE214982_extracted',
        'datasets': [
            {
                'name': 'mouse_adipose_progenitors',
                'files': {
                    'tar': 'GSE214982_RAW.tar'
                },
                'tissue': 'adipose',
                'cell_type': 'stromal_cells'
            }
        ]
    },
    # New dataset: GSE207705 - Cold-induced brown adipocyte neogenesis
    'GSE207705': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE207705_extracted',
        'datasets': [
            {
                'name': 'brown_adipose_timecourse',
                'files': {
                    'matrix': 'GSE207705_Quantseq_DataMatrix.csv.gz'
                },
                'tissue': 'brown_adipose',
                'cell_type': 'brown_adipocyte'
            }
        ]
    },
    # New dataset: GSE272039 - Mouse adipose single-cell data with resilience to obesity
    'GSE272039': {
        'species': 'mouse',
        'type': 'single-cell',
        'extracted_dir': 'GSE272039_extracted',
        'datasets': [
            {
                'name': 'adipose_obesity_response',
                'files': {
                    'h5ad': 'GSE272039_adipose_snRNA.h5ad'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            }
        ]
    }
}

# Datasets to use for training HUMAN model
HUMAN_GEO_DATASETS = {
    'GSE288785': {
        'species': 'human',
        'type': 'single-nuclei',
        'extracted_dir': 'GSE288785_temp_extraction',
        'datasets': [
            {
                'name': '916-DM',
                'files': {
                    'barcodes': 'GSM8775447_916-DM_barcodes.tsv.gz',
                    'features': 'GSM8775447_916-DM_features.tsv.gz',
                    'matrix': 'GSM8775447_916-DM_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': '916-AdipoB',
                'files': {
                    'barcodes': 'GSM8775448_916-AdipoB_barcodes.tsv.gz',
                    'features': 'GSM8775448_916-AdipoB_features.tsv.gz',
                    'matrix': 'GSM8775448_916-AdipoB_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': '49-DM',
                'files': {
                    'barcodes': 'GSM8775449_49-DM_barcodes.tsv.gz',
                    'features': 'GSM8775449_49-DM_features.tsv.gz',
                    'matrix': 'GSM8775449_49-DM_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            },
            {
                'name': '49-AdipoB',
                'files': {
                    'barcodes': 'GSM8775450_49-AdipoB_barcodes.tsv.gz',
                    'features': 'GSM8775450_49-AdipoB_features.tsv.gz',
                    'matrix': 'GSM8775450_49-AdipoB_matrix.mtx.gz'
                },
                'tissue': 'adipose',
                'cell_type': 'mixed'
            }
        ]
    },
    'GSE249089': {
        'species': 'human',
        'type': 'single-nuclei',
        'extracted_dir': '.',
        'datasets': [
            {
                'name': 'subcutaneous_84_finnish',
                'files': {
                    'barcodes': 'GSE249089_barcodes.tsv.gz',
                    'features': 'GSE249089_features.tsv.gz',
                    'matrix': 'GSE249089_matrix.mtx.gz',
                    'metadata': 'GSE249089_meta_data.tsv.gz'
                },
                'tissue': 'subcutaneous',
                'cell_type': 'mixed'
            }
        ]
    },
    'GSE236708': {
        'species': 'human',
        'type': 'single-nuclei',
        'extracted_dir': '.',
        'datasets': [
            {
                'name': 'subcutaneous_16_finnish',
                'files': {
                    'barcodes': 'GSE236708_barcodes.tsv.gz',
                    'features': 'GSE236708_features.tsv.gz',
                    'matrix': 'GSE236708_matrix.mtx.gz',
                    'metadata': 'GSE236708_meta_data.tsv.gz'
                },
                'tissue': 'subcutaneous',
                'cell_type': 'mixed'
            }
        ]
    },
    # Consider adding human datasets from the same studies if available
}

# Mouse cell markers (MOUSE_CELL_MARKERS)
MOUSE_CELL_MARKERS = {
    'white_adipocyte': ['Lep', 'Adipoq', 'Retn', 'Lpl', 'Fabp4', 'Plin1', 'Pparg', 'Cd36'],
    'beige_adipocyte': ['Tmem26', 'Tbx1', 'Cd137', 'Slc27a1', 'Pat2', 'Cited1', 'Ear2', 'Ppargc1a'],
    'brown_adipocyte': ['Ucp1', 'Cidea', 'Prdm16', 'Ppargc1a', 'Dio2', 'Cox8b', 'Elovl3', 'Adrb3', 'Asc1', 'Pgc1a'],
    'macrophages': ['Adgre1', 'Cd68', 'Itgam', 'Mrc1', 'Cd14', 'Fcgr1', 'Csf1r', 'Aif1'],
    'immune_cells': ['Ptprc', 'Cd3e', 'Cd4', 'Cd8a', 'Il6', 'Tnf', 'Il1b', 'Ifng'],
    'fibroblasts': ['Col1a1', 'Col3a1', 'Dcn', 'Pdgfra', 'Vim', 'Fbn1', 'Acta2', 'S100a4'],
    'vascular_cells': ['Pecam1', 'Cdh5', 'Vwf', 'Flt1', 'Kdr', 'Tie1', 'Nos3', 'Icam1'],
    'innate_T_cells': ['Il17a', 'Rorc', 'Il22', 'Stat3', 'Ccr6', 'Il23r', 'Il7r', 'Cd3e'],
    'stromal_cells': ['Pdgfra', 'Cd34', 'Ly6a', 'Cd29', 'Cd44', 'Cd140a', 'Sca1', 'Thy1', 'Bmper', 'Dpp4', 'Cd55', 'Cd142'],
    'adipose_progenitors': ['Bmper', 'Pdgfra', 'Cd34', 'Ly6a', 'Cd29', 'Dpp4', 'Cd55', 'Cd142']
}

# Human cell markers (HUMAN_CELL_MARKERS)
HUMAN_CELL_MARKERS = {
    'white_adipocyte': ['LEP', 'ADIPOQ', 'RETN', 'LPL', 'FABP4', 'PLIN1', 'PPARG', 'CD36'],
    'beige_adipocyte': ['TMEM26', 'TBX1', 'CD137', 'SLC27A1', 'PAT2', 'CITED1', 'EAR2', 'PPARGC1A'],
    'brown_adipocyte': ['UCP1', 'CIDEA', 'PRDM16', 'PPARGC1A', 'DIO2', 'COX8B', 'ELOVL3', 'PGC1A'],
    'fibroblasts': ['COL1A1', 'COL3A1', 'DCN', 'PDGFRA', 'VIM', 'FBN1', 'ACTA2', 'S100A4'],
    'immune_cells': ['PTPRC', 'CD3E', 'CD4', 'CD8A', 'IL6', 'TNF', 'IL1B', 'IFNG'],
    'endothelial_cells': ['PECAM1', 'CDH5', 'VWF', 'FLT1', 'KDR', 'TIE1', 'NOS3', 'ICAM1'],
    'vascular_smooth_muscle': ['ACTA2', 'TAGLN', 'MYH11', 'CNN1', 'CALD1', 'DES', 'MYOCD', 'SMTN'],
    'preadipocytes': ['PDGFRA', 'CD34', 'THY1', 'DPP4', 'CD29', 'CD44', 'CD140A', 'CD90', 'BMPER', 'CD55', 'CD142'],
    'adipose_progenitors': ['BMPER', 'PDGFRA', 'CD34', 'CD55', 'DPP4', 'CD142', 'CD29', 'THY1']
}

# ---------------------------
# ORIGINAL IMPORTS (UNCHANGED)
# ---------------------------
import os
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.manifold import TSNE
import joblib
import logging
from scipy import io, sparse
import anndata
import shutil
from collections import defaultdict
import json
import h5py
import tarfile
import tempfile
import re
import fnmatch
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# ---------------------------
# ORIGINAL CONFIGURATION (UNCHANGED)
# ---------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configure_scanpy():
    """Configure scanpy to avoid numba errors"""
    try:
        os.environ['NUMBA_DISABLE_JIT'] = '1'        
        sc.settings.verbosity = 0
        sc.settings.n_jobs = 1
        sc.settings.var_names_make_unique = True
        logger.info("Scanpy configuration successful")
    except Exception as e:
        logger.error(f"Error configuring scanpy: {e}")

# ---------------------------
# ORIGINAL PROCESSING FUNCTIONS (UNCHANGED)
# ---------------------------
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

def extract_tar_file(tar_path, extract_dir):
    """
    Extract a TAR file to the specified directory
    """
    logger.info(f"Extracting {tar_path} to {extract_dir}")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Create a safe extraction path
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract all files
            tar.extractall(path=extract_dir)
            
        logger.info(f"Successfully extracted {tar_path}")
        return True
    except Exception as e:
        logger.error(f"Error extracting {tar_path}: {e}")
        return False

def find_10x_files_in_dir(directory):
    """
    Find 10X Genomics format files in a directory (barcodes, features, matrix)
    """
    logger.info(f"Searching for 10X files in {directory}")
    
    # Patterns to match 10X files
    barcode_patterns = ['*barcodes*', '*cell*', '*barcode*']
    feature_patterns = ['*features*', '*genes*', '*gene*']
    matrix_patterns = ['*matrix*', '*mtx*']
    
    barcodes_file = None
    features_file = None
    matrix_file = None
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if file matches barcode patterns
            if any(fnmatch.fnmatch(file.lower(), pattern) for pattern in barcode_patterns) and not barcodes_file:
                barcodes_file = file_path
            
            # Check if file matches feature patterns
            elif any(fnmatch.fnmatch(file.lower(), pattern) for pattern in feature_patterns) and not features_file:
                features_file = file_path
            
            # Check if file matches matrix patterns
            elif any(fnmatch.fnmatch(file.lower(), pattern) for pattern in matrix_patterns) and not matrix_file:
                matrix_file = file_path
    
    # Return found files
    if barcodes_file and features_file and matrix_file:
        logger.info(f"Found 10X files: {os.path.basename(barcodes_file)}, {os.path.basename(features_file)}, {os.path.basename(matrix_file)}")
        return barcodes_file, features_file, matrix_file
    else:
        logger.warning(f"Could not find complete set of 10X files in {directory}")
        return None, None, None

def read_gzipped_file(file_path):
    """Read a gzipped file properly"""
    try:
        with gzip.open(file_path, 'rb') as f:
            content = f.read().decode('utf-8')
        return [line for line in content.split('\n') if line]
    except Exception as e:
        logger.error(f"Error reading gzipped file {file_path}: {e}")
        return []

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

def process_loom_data(loom_path, dataset_info):
    """
    Process Loom format data
    """
    name = dataset_info['name']
    logger.info(f"Processing Loom dataset: {name}")
    
    # Check if file exists
    if not os.path.exists(loom_path):
        logger.error(f"Missing Loom file for {name}: {loom_path}")
        return None
    
    try:
        # Load loom file
        adata = sc.read_loom(loom_path)
        logger.info(f"Successfully loaded {name} as Loom: {adata.shape[0]} cells, {adata.shape[1]} genes")
        
        # Add dataset metadata
        adata.obs['dataset'] = name
        adata.obs['tissue'] = dataset_info['tissue']
        adata.obs['initial_type'] = dataset_info['cell_type']
        
        return adata
    except Exception as e:
        logger.error(f"Error loading Loom data for {name}: {e}")
        return None

def process_rds_data(rds_path, dataset_info):
    """
    Process RDS format data (requires rpy2)
    """
    name = dataset_info['name']
    logger.info(f"Processing RDS dataset: {name}")
    
    # Check if file exists
    if not os.path.exists(rds_path):
        logger.error(f"Missing RDS file for {name}: {rds_path}")
        return None
    
    try:
        # Warning about rpy2 dependency
        logger.warning("RDS processing requires rpy2 package. If not installed, this will fail.")
        
        # Import rpy2 inside function to avoid dependency issues
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        
        # Read RDS
        r_data = ro.r['readRDS'](rds_path)
        
        # Convert to AnnData - this is a simplified approach and may need adjustment
        # based on actual RDS structure (Seurat, SingleCellExperiment, etc.)
        with localconverter(ro.default_converter + pandas2ri.converter):
            # Extract counts - adjust based on actual structure
            if hasattr(r_data, 'assays'):
                counts = ro.conversion.rpy2py(r_data.assays.data['counts'])
            else:
                counts = ro.conversion.rpy2py(r_data)
            
            # Create AnnData object
            adata = anndata.AnnData(X=counts)
            
        # Add dataset metadata
        adata.obs['dataset'] = name
        adata.obs['tissue'] = dataset_info['tissue']
        adata.obs['initial_type'] = dataset_info['cell_type']
        
        logger.info(f"Successfully loaded {name} as RDS: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
    except ImportError:
        logger.error("rpy2 package not installed. Cannot process RDS files.")
        return None
    except Exception as e:
        logger.error(f"Error loading RDS data for {name}: {e}")
        return None

# Function to add unique identifiers to observation and variable names
def make_names_unique(adata):
    """
    Make observation and variable names unique to avoid warnings and errors
    """
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata

def process_geo_dataset(geo_accession, dataset_dict, data_dir='geo_data', max_cells_per_dataset=5000):
    """
    Process datasets from a specific GEO accession
    """
    logger.info(f"Processing {geo_accession} data")
    
    if not geo_accession in dataset_dict:
        logger.error(f"No dataset info found for {geo_accession}")
        return None
    
    dataset_info = dataset_dict[geo_accession]
    
    # Handle different directory structures
    if dataset_info['extracted_dir'] == '.':
        # Files are directly in the data_dir
        extracted_dir = data_dir
    else:
        # Files are in a subdirectory
        extracted_dir = os.path.join(data_dir, dataset_info['extracted_dir'])
    
    # Create directory if it doesn't exist (for output directories)
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Process each dataset in the GEO accession
    adatas = []
    
    for ds in dataset_info['datasets']:
        # Special handling for 10X format files from GSE176171
        if 'counts_barcodes' in ds.get('files', {}):
            adata = process_10x_special_format(ds, extracted_dir)
        # Process based on file type
        elif 'h5' in ds.get('files', {}):
            h5_path = os.path.join(extracted_dir, ds['files']['h5'])
            adata = process_h5_data(h5_path, ds)
        elif 'loom' in ds.get('files', {}):
            loom_path = os.path.join(extracted_dir, ds['files']['loom'])
            adata = process_loom_data(loom_path, ds)
        elif 'rds' in ds.get('files', {}):
            rds_path = os.path.join(extracted_dir, ds['files']['rds'])
            adata = process_rds_data(rds_path, ds)
        elif all(key in ds.get('files', {}) for key in ['barcodes', 'features', 'matrix']):
            # Check if files exist
            barcodes_path = os.path.join(extracted_dir, ds['files']['barcodes'])
            features_path = os.path.join(extracted_dir, ds['files']['features'])
            matrix_path = os.path.join(extracted_dir, ds['files']['matrix'])
            
            if all(os.path.exists(p) for p in [barcodes_path, features_path, matrix_path]):
                logger.info(f"Files found for {ds['name']}:")
                logger.info(f"  Barcodes: {barcodes_path}")
                logger.info(f"  Features: {features_path}")
                logger.info(f"  Matrix: {matrix_path}")
                adata = process_10x_data(ds, extracted_dir)
            else:
                missing = [p for p in [barcodes_path, features_path, matrix_path] if not os.path.exists(p)]
                logger.error(f"Missing files for {ds['name']}: {missing}")
                adata = None
        elif 'tar' in ds.get('files', {}):
            # Handle tar file
            tar_path = os.path.join(extracted_dir, ds['files']['tar'])
            
            # Debug info
            logger.info(f"Looking for TAR file at: {tar_path}")
            if os.path.exists(tar_path):
                logger.info(f"TAR file found: {tar_path}")
            else:
                logger.error(f"TAR file not found: {tar_path}")
                continue
                
            # Create temp directory for extraction
            temp_dir = os.path.join(data_dir, f"{geo_accession}_temp_extraction")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract tar file
            if extract_tar_file(tar_path, temp_dir):
                logger.info(f"Successfully extracted {tar_path} to {temp_dir}")
                # Try to find and process 10X, H5, Loom, or RDS files
                barcodes_path, features_path, matrix_path = find_10x_files_in_dir(temp_dir)
                
                if all(p is not None for p in [barcodes_path, features_path, matrix_path]):
                    # Create temporary dataset info for 10X data
                    temp_ds = ds.copy()
                    temp_ds['files'] = {
                        'barcodes': barcodes_path,
                        'features': features_path,
                        'matrix': matrix_path
                    }
                    adata = process_10x_data(temp_ds, "")  # Empty base dir since we're providing full paths
                else:
                    # Look for other file types
                    h5_files = []
                    loom_files = []
                    rds_files = []
                    
                    # Walk through all subdirectories
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.h5') or file.endswith('.h5ad'):
                                h5_files.append(os.path.join(root, file))
                            elif file.endswith('.loom'):
                                loom_files.append(os.path.join(root, file))
                            elif file.endswith('.rds'):
                                rds_files.append(os.path.join(root, file))
                    
                    logger.info(f"Found {len(h5_files)} H5 files, {len(loom_files)} Loom files, {len(rds_files)} RDS files")
                    
                    if h5_files:
                        h5_path = h5_files[0]
                        logger.info(f"Using H5 file: {h5_path}")
                        adata = process_h5_data(h5_path, ds)
                    elif loom_files:
                        loom_path = loom_files[0]
                        logger.info(f"Using Loom file: {loom_path}")
                        adata = process_loom_data(loom_path, ds)
                    elif rds_files:
                        rds_path = rds_files[0]
                        logger.info(f"Using RDS file: {rds_path}")
                        adata = process_rds_data(rds_path, ds)
                    else:
                        logger.error(f"No supported files found in extracted tar for {ds['name']}")
                        adata = None
            else:
                logger.error(f"Failed to extract TAR file: {tar_path}")
                adata = None
        else:
            logger.error(f"Unsupported data format for {ds['name']}")
            adata = None
        
        if adata is not None:
            # Basic preprocessing
            logger.info(f"Running basic preprocessing on {ds['name']}")
            
            # Make variable names unique
            adata.var_names_make_unique()
            adata.obs_names_make_unique()
            
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
            # Make sure names are unique after concatenation
            combined_adata.var_names_make_unique()
            combined_adata.obs_names_make_unique()
            
            logger.info(f"Combined {len(adatas)} datasets for {geo_accession}: {combined_adata.shape[0]} cells, {combined_adata.shape[1]} genes")
            return combined_adata
        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            # Fall back to returning the first dataset
            logger.info(f"Falling back to using only the first dataset")
            return adatas[0]
    else:
        return adatas[0]

def process_10x_data(dataset_info, base_dir):
    """
    Process 10X Genomics format data (matrix, features, barcodes)
    """
    name = dataset_info['name']
    files = dataset_info['files']
    logger.info(f"Processing 10X dataset: {name}")
    
    # Construct file paths
    barcodes_path = os.path.join(base_dir, files.get('barcodes', ''))
    features_path = os.path.join(base_dir, files.get('features', ''))
    matrix_path = os.path.join(base_dir, files.get('matrix', ''))
    
    # Check if files exist
    if not all(p and os.path.exists(p) for p in [barcodes_path, features_path, matrix_path]):
        missing = [p for p in [barcodes_path, features_path, matrix_path] if not p or not os.path.exists(p)]
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
        
        # Make names unique
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        
        logger.info(f"Successfully loaded {name}: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
    
    except Exception as e:
        logger.error(f"Error loading 10X data for {name}: {e}")
        return None

def process_10x_special_format(dataset_info, base_dir):
    """
    Process 10X format data that has separate files for barcodes, features, and matrix
    but with non-standard file naming (e.g., GSE176171)
    """
    name = dataset_info['name']
    files = dataset_info['files']
    logger.info(f"Processing special 10X dataset: {name}")
    
    # Determine if this is the special format (GSE176171)
    is_special_format = 'counts_barcodes' in files and 'counts_features' in files and 'counts_mtx' in files
    
    if is_special_format:
        # Construct file paths for special format
        barcodes_path = os.path.join(base_dir, files['counts_barcodes'])
        features_path = os.path.join(base_dir, files['counts_features'])
        matrix_path = os.path.join(base_dir, files.get('counts_mtx', ''))
        
        # Debug logging
        logger.info(f"Looking for special format files at:")
        logger.info(f"  Barcodes: {barcodes_path}")
        logger.info(f"  Features: {features_path}")
        logger.info(f"  Matrix: {matrix_path}")
    else:
        # Standard format
        barcodes_path = os.path.join(base_dir, files.get('barcodes', ''))
        features_path = os.path.join(base_dir, files.get('features', ''))
        matrix_path = os.path.join(base_dir, files.get('matrix', ''))
    
    # Check if files exist
    if not all(p and os.path.exists(p) for p in [barcodes_path, features_path, matrix_path]):
        missing = [p for p in [barcodes_path, features_path, matrix_path] if not p or not os.path.exists(p)]
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
        
        # Make names unique
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        
        # Read metadata if available
        if 'metadata' in files:
            metadata_path = os.path.join(base_dir, files['metadata'])
            if os.path.exists(metadata_path):
                logger.info(f"Reading metadata file: {metadata_path}")
                try:
                    metadata = pd.read_csv(metadata_path, sep='\t', index_col=0)
                    # Adjust index to match the AnnData object if needed
                    common_indices = adata.obs.index.intersection(metadata.index)
                    if len(common_indices) > 0:
                        # Add metadata columns to AnnData.obs
                        for col in metadata.columns:
                            adata.obs[col] = metadata.loc[common_indices, col]
                    else:
                        logger.warning(f"No common indices between AnnData and metadata for {name}")
                except Exception as e:
                    logger.error(f"Error reading metadata for {name}: {e}")
        
        logger.info(f"Successfully loaded {name}: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
    
    except Exception as e:
        logger.error(f"Error loading special 10X data for {name}: {e}")
        return None

# ---------------------------
# UPDATED PREPROCESSING WITH HIERARCHICAL CLUSTERING
# ---------------------------
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD

def preprocess_and_cluster(adata, n_clusters=20, species='mouse'):
    """Process AnnData object with manual K-means clustering"""
    logger.info(f"Running manual K-means clustering for {species}")
    
    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Manual processing
    manual_highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]
    manual_pca(adata)
    manual_tsne(adata)
    
    # Run K-means clustering
    manual_kmeans_clustering(adata, n_clusters=n_clusters)
    
    return adata
# ---------------------------
# UPDATED ANNOTATION SYSTEM
# ---------------------------
def annotate_clusters(adata, marker_dict, species='mouse'):
    """Annotate clusters based on marker gene expression"""
    logger.info(f"Annotating clusters for {species}")
    
    cluster_scores = {}
    
    # Only use kmeans clusters
    cluster_key = 'kmeans'
    
    for cluster in adata.obs[cluster_key].cat.categories:
        cluster_cells = adata[adata.obs[cluster_key] == cluster]
        scores = {}
        
        for cell_type, markers in marker_dict.items():
            markers_in_data = [m for m in markers if m in adata.var_names]
            if not markers_in_data:
                continue
                
            # Calculate mean expression safely
            if sparse.issparse(cluster_cells.X):
                expr = cluster_cells[:, markers_in_data].X.mean(axis=0).A1
            else:
                expr = cluster_cells[:, markers_in_data].X.mean(axis=0)
                
            scores[cell_type] = float(np.mean(expr))
        
        cluster_scores[cluster] = scores
    
    # Store annotations
    adata.uns['cluster_annotations'] = cluster_scores
    
    # Assign cell types based on clusters
    cell_types = []
    for idx in range(adata.n_obs):
        cluster = adata.obs[cluster_key][idx]
        scores = cluster_scores[cluster]
        if scores:
            cell_type = max(scores, key=scores.get)
        else:
            cell_type = 'unknown'
        cell_types.append(cell_type)
    
    adata.obs['cell_type'] = cell_types
    
    return adata

def calculate_consensus_annotation(adata):
    """Updated for hierarchical keys"""
    consensus = []
    resolutions = sorted(adata.uns['cluster_hierarchy'].keys(), key=lambda x: float(x.split('_')[1]))
    
    for cell in adata.obs.index:
        votes = []
        for res in resolutions:
            cluster = adata.obs[res][cell]
            ann = adata.uns['hierarchical_annotations'][res][cluster]
            votes.append(max(ann, key=ann.get) if ann else 'unknown')
        consensus.append(max(set(votes), key=votes.count))
    
    return consensus

# ---------------------------
# UPDATED VISUALIZATION
# ---------------------------
def visualize_results(adata, output_dir='.', species='mouse'):
    """Visualize clustering results focusing on K-means"""
    fig_dir = os.path.join(output_dir, 'figures', species)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plot K-means clusters
    try:
        plt.figure(figsize=(10, 8))
        sc.pl.tsne(adata, color='kmeans', legend_loc='on data', show=False)
        plt.title(f"{species.capitalize()} K-means Clusters (n={len(adata.obs['kmeans'].cat.categories)})")
        plt.savefig(os.path.join(fig_dir, f'{species}_kmeans_clusters.png'))
        plt.close()
    except Exception as e:
        logger.error(f"Error visualizing K-means clusters: {e}")
    
    # Plot cell types
    try:
        plt.figure(figsize=(10, 8))
        sc.pl.tsne(adata, color='cell_type', legend_loc='on data', show=False)
        plt.title(f"{species.capitalize()} Cell Types")
        plt.savefig(os.path.join(fig_dir, f'{species}_cell_types.png'))
        plt.close()
    except Exception as e:
        logger.error(f"Error visualizing cell types: {e}")
        
    # Create cluster-cell type relationship visualization
    try:
        # Create a cross-tabulation between clusters and cell types
        cluster_celltype = pd.crosstab(
            adata.obs['kmeans'], 
            adata.obs['cell_type'],
            normalize='index'  # Normalize by row (cluster)
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cluster_celltype, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f"{species.capitalize()} Cluster to Cell Type Mapping")
        plt.xlabel('Cell Type')
        plt.ylabel('K-means Cluster')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{species}_cluster_celltype_mapping.png'))
        plt.close()
        
        # Save the mapping to CSV for reference
        cluster_celltype.to_csv(os.path.join(fig_dir, f'{species}_cluster_celltype_mapping.csv'))
    except Exception as e:
        logger.error(f"Error creating cluster-cell type mapping: {e}")
    
    return
    
def plot_marker_violins(adata, fig_dir, species):
    """Plot violin plots for marker genes"""
    # Check available markers
    all_markers = ['Ucp1', 'Lep', 'Adipoq', 'Cd36', 'Pparg', 'Fabp4']  # Extended adipocyte markers
    available_markers = [m for m in all_markers if m in adata.var_names]
    
    if not available_markers:
        logger.warning(f"No adipocyte markers found in dataset. Available genes: {list(adata.var_names[:5])}...")
        return
    
    logger.info(f"Plotting violin plots for markers: {available_markers}")
    
    plt.figure(figsize=(12, 6))
    sc.pl.violin(
        adata,
        keys=available_markers[:3],  # Use up to 3 available markers
        groupby='kmeans',
        rotation=45,
        show=False
    )
    plt.savefig(os.path.join(fig_dir, f'{species}_marker_violins.png'))
    plt.close()

def plot_annotation_consistency(adata, output_dir):
    """Updated for hierarchical clustering"""
    # Get two most granular resolutions
    resolutions = sorted(
        [key for key in adata.obs.columns if key.startswith('hierarchical_')],
        key=lambda x: float(x.split('_')[1])
    )[-2:]  # Last two resolutions
    
    if len(resolutions) < 2:
        logger.warning("Not enough resolutions for consistency plot")
        return
    
    parent_key, child_key = resolutions
    consistency = pd.crosstab(adata.obs[parent_key], adata.obs[child_key])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(consistency, annot=True, fmt='d', cmap='viridis')
    plt.title(f"Cluster consistency: {parent_key} vs {child_key}")
    plt.savefig(os.path.join(output_dir, 'annotation_consistency.png'))
    plt.close()

# ---------------------------
# ORIGINAL MODEL FUNCTIONS WITH ENHANCEMENTS
# ---------------------------
def prepare_features_for_training(adata, cell_markers, species='mouse'):
    """
    Prepare features for training the random forest model from annotated data
    """
    logger.info(f"Preparing features for training {species} model")
    
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
        for ct, markers in cell_markers.items():
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
        
        # Add cluster feature
        feature_dict['kmeans_cluster'] = int(adata.obs['kmeans'][cell_idx])
        
        # Get label (true cell type)
        cell_type = adata.obs['cell_type'][cell_idx]
        
        # Add to lists
        features.append(feature_dict)
        labels.append(cell_type)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    
    logger.info(f"Prepared {len(feature_df)} {species} cells with {len(feature_df.columns)} features")
    
    return feature_df, np.array(labels)

def train_random_forest_model(features, labels, species='mouse', output_dir='models'):
    """Train a Random Forest model with confusion matrix generation"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    # Species-specific config
    config = {
        'mouse': {'n_estimators': 200, 'max_depth': 15},
        'human': {'n_estimators': 300, 'max_depth': 20}
    }
    params = config.get(species.lower(), config['mouse'])
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Accuracy: {accuracy:.2f}")
    logger.info(f"Feature importances:\n{pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)}")
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique class labels
    class_labels = np.unique(np.concatenate([y_test, y_pred]))
    
    # Plot confusion matrix
    fig_dir = os.path.join(output_dir, 'figures', species)
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{species.capitalize()} Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save confusion matrix
    confusion_matrix_path = os.path.join(fig_dir, f'{species}_confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
    
    # Generate detailed classification report
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")
    
    # Save classification report to file
    report_path = os.path.join(fig_dir, f'{species}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: {species.capitalize()} Adipocyte Classifier\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    return model

def calculate_cluster_stability(row, adata):
    """Updated for hierarchical clustering"""
    # Get sorted resolutions from cluster hierarchy
    resolutions = sorted(
        [key for key in adata.obs.columns if key.startswith('hierarchical_')],
        key=lambda x: float(x.split('_')[1])
    )  # Close parenthesis added here
    
    # Need at least 3 resolutions for stability calculation
    if len(resolutions) < 3:
        return 0
    
    clusters = [
        adata.obs.loc[row.name, resolutions[0]],
        adata.obs.loc[row.name, resolutions[1]],
        adata.obs.loc[row.name, resolutions[2]]
    ]
    return len(set(clusters))
# ---------------------------
# REMAINING ORIGINAL FUNCTIONS (UNCHANGED)
# ---------------------------
def save_model_and_artifacts(model, feature_names, output_dir='models', species='mouse'):
    """
    Save the trained model and related artifacts
    """
    # Create species-specific directory
    model_dir = os.path.join(output_dir, species)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f'adipocyte_annotator_{species}_model.joblib')
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_path = os.path.join(model_dir, 'feature_names.txt')
    with open(feature_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    # Save cell markers (using the appropriate marker set from the global variables)
    marker_path = os.path.join(model_dir, 'cell_markers.json')
    with open(marker_path, 'w') as f:
        if species.lower() == 'mouse':
            json.dump(MOUSE_CELL_MARKERS, f, indent=2)
        else:
            json.dump(HUMAN_CELL_MARKERS, f, indent=2)
    
    logger.info(f"{species.capitalize()} model and artifacts saved to {model_dir}")
    return model_path

def process_datasets_and_train_model(dataset_dict, data_dir='geo_data', 
                                   output_dir='models', species='mouse',
                                   n_clusters=20):
    """Pipeline with manual K-means clustering"""
    logger.info(f"Starting {species} model training pipeline")
    
    adatas = []
    for geo_id in dataset_dict:
        # Process GEO dataset
        adata = process_geo_dataset(
            geo_id, 
            dataset_dict, 
            data_dir=data_dir,
            max_cells_per_dataset=5000
        )
        
        if adata is not None:
            # Preprocessing with manual K-means
            adata = preprocess_and_cluster(
                adata,
                n_clusters=n_clusters,
                species=species
            )
            
            # Annotation
            if species.lower() == 'mouse':
                adata = annotate_clusters(adata, MOUSE_CELL_MARKERS, species=species)
            else:
                adata = annotate_clusters(adata, HUMAN_CELL_MARKERS, species=species)
            
            # Visualization
            visualize_results(adata, output_dir, species=species)
            
            # Plot marker violins
            fig_dir = os.path.join(output_dir, 'figures', species)
            plot_marker_violins(adata, fig_dir, species)
            
            adatas.append(adata)
    
    if not adatas:
        logger.error(f"No valid datasets processed for {species}")
        return None, None
    
    # Dataset combination
    if len(adatas) > 1:
        # Add unique obs names to avoid conflicts
        for i, adata in enumerate(adatas):
            adata.obs_names_make_unique(f"dataset{i}_")
        
        combined_adata = sc.concat(adatas, join='outer')
        # Make sure names are unique after concatenation
        combined_adata.var_names_make_unique()
        combined_adata.obs_names_make_unique()
    else:
        combined_adata = adatas[0]
    
    # Prepare features
    if species == 'mouse':
        features, labels = prepare_features_for_training(
            combined_adata, 
            MOUSE_CELL_MARKERS,
            species=species
        )
    else:
        features, labels = prepare_features_for_training(
            combined_adata,
            HUMAN_CELL_MARKERS,
            species=species
        )
    
    # Train model with confusion matrix generation
    logger.info(f"Training {species} model")
    model = train_random_forest_model(features, labels, species=species, output_dir=output_dir)
    
    # Save model
    model_path = save_model_and_artifacts(
        model,
        features.columns,
        output_dir=output_dir,
        species=species
    )
    
    # Cross-validation to ensure robustness
    logger.info(f"Performing cross-validation for {species} model")
    cv_scores = cross_val_score(model, features, labels, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Save cross-validation results
    cv_path = os.path.join(output_dir, 'figures', species, f'{species}_cv_results.txt')
    with open(cv_path, 'w') as f:
        f.write(f"Cross-validation results for {species} model:\n")
        f.write(f"Individual scores: {cv_scores}\n")
        f.write(f"Mean accuracy: {cv_scores.mean():.4f}\n")
        f.write(f"Standard deviation: {cv_scores.std():.4f}\n")
    
    logger.info(f"{species} pipeline completed successfully")
    return model, combined_adata

def predict_cell_types(adata, model_path, cell_markers=None, species='mouse'):
    """
    Predict cell types for a new dataset using a trained model
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with expression data
    model_path : str
        Path to the trained model file (.joblib)
    cell_markers : dict, optional
        Dictionary mapping cell types to marker genes
    species : str
        'mouse' or 'human'
        
    Returns:
    --------
    adata : AnnData
        AnnData object with cell type predictions added
    """
    logger.info(f"Predicting cell types for {species} dataset with {adata.shape[0]} cells")
    
    # Load model
    model = joblib.load(model_path)
    
    # Get feature names
    model_dir = os.path.dirname(model_path)
    feature_path = os.path.join(model_dir, 'feature_names.txt')
    
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        logger.warning(f"Feature names file not found at {feature_path}. Using model classes.")
        feature_names = None
    
    # Load cell markers if not provided
    if cell_markers is None:
        marker_path = os.path.join(model_dir, 'cell_markers.json')
        if os.path.exists(marker_path):
            with open(marker_path, 'r') as f:
                cell_markers = json.load(f)
        else:
            logger.warning(f"Cell markers file not found at {marker_path}. Using default markers.")
            if species.lower() == 'mouse':
                cell_markers = MOUSE_CELL_MARKERS
            else:
                cell_markers = HUMAN_CELL_MARKERS
    
    # Prepare features
    features_df, _ = prepare_features_for_training(adata, cell_markers, species=species)
    
    # Select features used by the model if feature_names is available
    if feature_names is not None:
        # Check which features are available
        available_features = [f for f in feature_names if f in features_df.columns]
        
        if len(available_features) < len(feature_names):
            logger.warning(f"Only {len(available_features)}/{len(feature_names)} model features are available.")
        
        if not available_features:
            logger.error("No model features available in the input data. Cannot predict.")
            return adata
        
        features_df = features_df[available_features]
    
    # Predict cell types
    predictions = model.predict(features_df)
    prediction_proba = model.predict_proba(features_df)
    
    # Add predictions to AnnData object
    adata.obs['predicted_cell_type'] = predictions
    adata.obs['predicted_cell_type'] = adata.obs['predicted_cell_type'].astype('category')
    
    # Add confidence scores (max probability)
    adata.obs['prediction_confidence'] = np.max(prediction_proba, axis=1)
    
    # Add probabilities for each cell type
    for i, cell_type in enumerate(model.classes_):
        adata.obs[f'prob_{cell_type}'] = prediction_proba[:, i]
    
    # Show distribution of predicted cell types
    pred_type_counts = adata.obs['predicted_cell_type'].value_counts()
    logger.info(f"Predicted cell type distribution for {species}:")
    for cell_type, count in pred_type_counts.items():
        logger.info(f"  {cell_type}: {count} cells ({count/adata.shape[0]*100:.1f}%)")
    
    return adata

def predict_with_both_models(mouse_data_path, human_data_path, mouse_model_path=None, human_model_path=None, output_dir='predictions'):
    """
    Predict cell types for new datasets using both mouse and human models
    
    Parameters:
    -----------
    mouse_data_path : str
        Path to mouse dataset (h5ad, 10X, etc.)
    human_data_path : str
        Path to human dataset (h5ad, 10X, etc.)
    mouse_model_path : str, optional
        Path to mouse model. If None, will look in default location.
    human_model_path : str, optional
        Path to human model. If None, will look in default location.
    output_dir : str
        Directory to save prediction results
        
    Returns:
    --------
    mouse_adata : AnnData
        Mouse AnnData object with predictions
    human_adata : AnnData
        Human AnnData object with predictions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default model paths
    if mouse_model_path is None:
        mouse_model_path = 'models/mouse/adipocyte_annotator_mouse_model.joblib'
    
    if human_model_path is None:
        human_model_path = 'models/human/adipocyte_annotator_human_model.joblib'
    
    # Check if models exist
    mouse_model_exists = os.path.exists(mouse_model_path)
    human_model_exists = os.path.exists(human_model_path)
    
    if not mouse_model_exists:
        logger.error(f"Mouse model not found at {mouse_model_path}")
    
    if not human_model_exists:
        logger.error(f"Human model not found at {human_model_path}")
    
    # Load and predict with mouse model
    mouse_adata = None
    if mouse_model_exists and mouse_data_path:
        # Load mouse data
        logger.info(f"Loading mouse data from {mouse_data_path}")
        
        # Different loading based on file type
        if mouse_data_path.endswith('.h5ad'):
            mouse_adata = sc.read_h5ad(mouse_data_path)
        else:
            # Try to load as 10X data
            try:
                mouse_adata = sc.read_10x_mtx(mouse_data_path)
            except Exception as e:
                logger.error(f"Failed to load mouse data: {e}")
        
        if mouse_adata is not None:
            # Basic preprocessing (normalization, log transform)
            sc.pp.filter_cells(mouse_adata, min_genes=200)
            sc.pp.filter_genes(mouse_adata, min_cells=3)
            sc.pp.normalize_total(mouse_adata, target_sum=1e4)
            sc.pp.log1p(mouse_adata)
            
            # Predict cell types
            mouse_adata = predict_cell_types(mouse_adata, mouse_model_path, species='mouse')
            
            # Save predictions
            mouse_output_path = os.path.join(output_dir, 'mouse_predictions.h5ad')
            mouse_adata.write(mouse_output_path)
            logger.info(f"Mouse predictions saved to {mouse_output_path}")
    
    # Load and predict with human model
    human_adata = None
    if human_model_exists and human_data_path:
        # Load human data
        logger.info(f"Loading human data from {human_data_path}")
        
        # Different loading based on file type
        if human_data_path.endswith('.h5ad'):
            human_adata = sc.read_h5ad(human_data_path)
        else:
            # Try to load as 10X data
            try:
                human_adata = sc.read_10x_mtx(human_data_path)
            except Exception as e:
                logger.error(f"Failed to load human data: {e}")
        
        if human_adata is not None:
            # Basic preprocessing (normalization, log transform)
            sc.pp.filter_cells(human_adata, min_genes=200)
            sc.pp.filter_genes(human_adata, min_cells=3)
            sc.pp.normalize_total(human_adata, target_sum=1e4)
            sc.pp.log1p(human_adata)
            
            # Predict cell types
            human_adata = predict_cell_types(human_adata, human_model_path, species='human')
            
            # Save predictions
            human_output_path = os.path.join(output_dir, 'human_predictions.h5ad')
            human_adata.write(human_output_path)
            logger.info(f"Human predictions saved to {human_output_path}")
    
    return mouse_adata, human_adata
def main():
    """Main function with manual K-means clustering"""
    configure_scanpy()
    
    # Create output directories
    os.makedirs('models/mouse', exist_ok=True)
    os.makedirs('models/human', exist_ok=True)
    os.makedirs('models/figures/mouse', exist_ok=True) 
    os.makedirs('models/figures/human', exist_ok=True)
    
    # Mouse processing with manual K-means
    #logger.info("\n🚀 Starting Mouse Pipeline")
    #mouse_model, mouse_adata = process_datasets_and_train_model(
    #    MOUSE_GEO_DATASETS,
    #   data_dir='geo_data',
    #    output_dir='models',
    #    species='mouse',
    #    n_clusters=25
    #)
    
    # Human processing with manual K-means
    logger.info("\n🚀 Starting Human Pipeline")
    human_model, human_adata = process_datasets_and_train_model(
        HUMAN_GEO_DATASETS,
        data_dir='geo_data',
        output_dir='models',
        species='human',
        n_clusters=35
    )
    
    logger.info("\n✅ Pipeline completed successfully")
    return mouse_model, human_model

if __name__ == "__main__":
    main()