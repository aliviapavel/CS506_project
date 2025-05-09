# **Automated Adipocyte Cell Identifier and Annotator** #
A machine learning toolkit for classifying adipocyte subtypes from single-cell RNA sequencing data.
[https://youtu.be/Z2oN9tKxrFg]
## Testing
Clone this repo, then you can run this from the main directory:
```
pip install pytest
pip install -r requirements.txt
pytest tests/test_script.py -v -p no:warnings --disable-warnings
ls -lh tests/test_results/*.png
```
Then, look in the figures folder. The mouse model performs much better as it is more robust than the human one with available samples. The mouse model was my primary focus, as it has the most applicability and available data.
## Usage Examples
**Processing 10X Data**
```
import scanpy as sc
from adipocyte_classifier import preprocessing

# Load 10X data
adata = preprocessing.load_10x_data("path/to/10x/data", species="mouse")

# Basic preprocessing
adata = preprocessing.preprocess_data(adata)

# Save processed data
adata.write("processed_data.h5ad")
```
**Clasifying Cells**
```
from adipocyte_classifier import models

# Load processed data
adata = sc.read("processed_data.h5ad")

# Predict cell types
adata = models.predict(adata, species="mouse")

# Save results
adata.write("predictions.h5ad")
```
**Visualization**
```
from adipocyte_classifier import visualization

# Load predicted data
adata = sc.read("predictions.h5ad")

# Create UMAP plot
visualization.plot_umap(adata, color="predicted_cell_type", 
                       save_path="umap_plot.png")

# Plot marker gene expression
visualization.plot_markers(adata, markers=["Ucp1", "Pparg", "Adipoq"],
                          save_path="marker_expression.png")

# Create summary statistics
visualization.plot_cell_type_distribution(adata, save_path="cell_types.png")
```
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

-Fork the repository

-Create your feature branch (git checkout -b feature/amazing-feature)

-Commit your changes (git commit -m 'Add some amazing feature')

-Push to the branch (git push origin feature/amazing-feature)

-Open a Pull Request

## **Description**
For the past year, I have worked under Dr. Nabil Rabhi in the Biochemistry Department over at the BU Medical Campus. There I have learned the basics of the single-cell RNA sequencing pipeline, doing quality control, annotation, and now analysis. Through this process I found there are several rather tedious steps in the process that have the potential to be automated with specialized packages and models. There exist several annotation packages in which you can train the model with self-provided data sets. However, very few packages exist that are pre-trained. My proposed package would be one such pre-trained model, specialized on adipose tissue. This model would then be able to be used in my, as well as others, analysis of adipose tissue, allowing for an ease of identification and annotation of sequenced data.
#### Cell Type Annotator
This annotator is able to take an input data set, cluster at various resolutions [see Heirarchical Annotation subheading], identify most likely cell type by cluster based on existing knowledge of cell markers, and identify potential new cell types or pathways. This model is written, and intended to be used in Python, although there is potential for cross-platform use.
#### Single-Cell Annotation Process - For Those Who are Unfamiliar
For those unfamiliar with the single-cell annotation process, for each cluster the top N markers are looked at (usually N is around 5, but can vary depending on the context and case); there is no unanimous cell marker database, rather cell markers come from the compilation of markers identified in existing studies and publications. The current most comprehensive database is CellMarker 2.0[https://academic.oup.com/nar/article/51/D1/D870/6775381]. The top N markers are then compared to existing known markers in an attempt to identify the cell type. Often these markers will not match exactly, and often only a fraction of the known markers may be present in a cluster. These unrecognized markers, markers that are among the top N expressed but have not been identified in previous publications, may be indicative of either new cell (sub)types or new pathways. Cell pathways are a series of actions among molecules in a cell that leads to a certain product or a change in the cell (National Human Genome Research Institute, NHGRI). Essentially, unknown markers may be indicative of this change of cell state.
## **Potential Future Expansion:**
#### Heirarchical Annotation
As cells can have different levels of classification (e.g. Adipocytes, white/brown/beige adipocyte, pre-adipocyte, etc.) heirarchical annotation lends itself to the process of understanding how individual cells change and interact with a larger group. While heirarchical annotatin may initially seem complex, it is little more than annotating the dataset at different resolutions and identifying which clusters at a lower resolution give way to more specific and refined clusters at higher resolution. Future implementations fo the package could utilize the trained model to predict the most likely markers at each relevant resolution.
#### Identification of Unknown/New cell types and pathways
As alluded to in the previous description, I aim to train a model which can aid in identifying unknown or new cell types and pathways. I propose this be done by utilizing the confidence of each prediction. When predicting the cell type of a cluster from known markers, the model would report how closely a cluster fits to known cell-type markers (how many known markers are present, their rank of expression in top N markers, proportion of cells expressing known markers, etc.). I see this identification being most prominent with the heirarchical annotation. Take for example a dataset where, at a low resolution, a known cell-type cluster splits into several smaller clusters at a higher resolution. I plan to implement this model to display the most likely cell type(s) for each sub-cluster. I am not certain on the exact method I would use to compute and predict the likelihood of each cell type, but that is to be explored and refined in this class. 
## **Current State**
**Key Technical Features**:
- Dual-species support (Mouse/Human)
- Random Forest classifier with marker-based features
- Confidence scoring for predictions

## Datasets Processed

### Mouse Models
| GEO Accession | Tissue Types | Cell Types | Cells Processed |
|---------------|--------------|------------|-----------------|
| GSE272938     | eWAT, iWAT   | White adipocytes, Stromal | 20,000 |
| GSE266326     | Adipose      | Immune cells, Mixed | 15,000 |
| GSE280109     | Adipose      | Innate T cells | 5,000 |
| GSE236579     | Adipose      | Mixed       | 10,000 |
| GSE214982     | Adipose      | Progenitors | 7,500 |
| GSE207705     | Brown adipose| Brown adipocytes | 2,300 |
| GSE272039     | Adipose      | Obesity response | 8,000 |

### Human Models
| GEO Accession | Tissue       | Cell Types | Cells Processed |
|---------------|--------------|------------|-----------------|
| GSE288785     | Subcutaneous | Mixed       | 12,000 |
| GSE249089     | Subcutaneous | Mixed       | 9,500 |
| GSE236708     | Subcutaneous | Mixed       | 6,800 |

## Core Pipeline Architecture
```mermaid
graph TD
    A[Raw GEO Data] --> B{Format Detection}
    B -->|10X| C[MTX/TSV Loader]
    B -->|H5| D[AnnData Converter]
    C --> E[Preprocessing]
    D --> E
    E --> F[Marker-Based Annotation]
    F --> G[Visualization Suite]
