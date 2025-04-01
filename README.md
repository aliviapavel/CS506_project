# **Automated Adipocyte Cell Identifier and Annotator** #
[https://youtu.be/jer6YBh_jN4]
## **Description**
For the past year, I have worked under Dr. Nabil Rabhi in the Biochemistry Department over at the BU Medical Campus. There I have learned the basics of the single-cell RNA sequencing pipeline, doing quality control, annotation, and now analysis. Through this process I found there are several rather tedious steps in the process that have the potential to be automated with specialized packages and models. There exist several annotation packages in which you can train the model with self-provided data sets. However, very few packages exist that are pre-trained. My proposed package would be one such pre-trained model, specialized on adipose tissue. This model would then be able to be used in my, as well as others, analysis of adipose tissue, allowing for an ease of identification and annotation of sequenced data.
#### Cell Type Annotator
My proposed annotator would be a model trained with existing datasets. The model will be able to take an input data set, identify ideal clustering resolution(s) [see Heirarchical Annotation subheading], identify most likely cell type by cluster based on existing knowledge of cell markers, and identify potential new cell types or pathways. 
#### Single-Cell Annotation Process - For Those Who are Unfamiliar
For those unfamiliar with the single-cell annotation process, for each cluster the top N markers are looked at (usually N is around 5, but can vary depending on the context and case); there is no unanimous cell marker database, rather cell markers come from the compilation of markers identified in existing studies and publications. The current most comprehensive database is CellMarker 2.0[https://academic.oup.com/nar/article/51/D1/D870/6775381]. The top N markers are then compared to existing known markers in an attempt to identify the cell type. Often these markers will not match exactly, and often only a fraction of the known markers may be present in a cluster. These unrecognized markers, markers that are among the top N expressed but have not been identified in previous publications, may be indicative of either new cell (sub)types or new pathways. Cell pathways are a series of actions among molecules in a cell that leads to a certain product or a change in the cell (National Human Genome Research Institute, NHGRI). Essentially, unknown markers may be indicative of this change of cell state.
#### Heirarchical Annotation
As cells can have different levels of classification (e.g. Adipocytes, white/brown/beige adipocyte, pre-adipocyte, etc.) heirarchical annotation lends itself to the process of understanding how individual cells change and interact with a larger group. While heirarchical annotatin may initially seem complex, it is little more than annotating the dataset at different resolutions and identifying which clusters at a lower resolution give way to more specific and refined clusters at higher resolution. This part of my package would simply utilize the trained model to predict the most likely markers at each relevant resolution. I would also like to implement some method to help identify which resolutions are key resolutions, but heirarchical annotion is absouletly possible without this step.
#### Identification of Unknown/New cell types and pathways
As alluded to in the previous description, I aim to train a model which can aid in identifying unknown or new cell types and pathways. I propose this be done by utilizing the confidence of each prediction. When predicting the cell type of a cluster from known markers, the model would report how closely a cluster fits to known cell-type markers (how many known markers are present, their rank of expression in top N markers, proportion of cells expressing known markers, etc.). I see this identification being most prominent with the heirarchical annotation. Take for example a dataset where, at a low resolution, a known cell-type cluster splits into several smaller clusters at a higher resolution. I plan to implement this model to display the most likely cell type(s) for each sub-cluster. I am not certain on the exact method I would use to compute and predict the likelihood of each cell type, but that is to be explored and refined in this class. 
## **Current Progress**
### Data Processing
I have successfully implemented a data processing pipeline that:

- Reads and processes single-cell RNA-seq data from multiple GEO datasets (currently GSE272938 and GSE266326)
- Handles both 10X Genomics formats (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz) and H5 files
- Performs basic quality control (filtering cells with <200 genes, filtering genes present in <3 cells)
- Normalizes expression data (target_sum=1e4) and applies log transformation
- Identifies highly variable genes for dimensionality reduction

### Model Development
I have developed a Random Forest classification model to automatically annotate cell types in adipose tissue:

1. **Feature Engineering:** For each cell, I extract features based on the expression of marker genes for each adipocyte cell type

- Mean expression of marker genes for each cell type
- Maximum expression of marker genes for each cell type
- Proportion of expressed marker genes for each cell type
- General cell features (total gene counts, number of expressed genes)


2. **Model Training:**

- Trained on 8,607 cells from multiple adipose tissue sources
- 5-fold cross-validation with a mean score of 0.76
- Overall accuracy on test set: 83%


3. **Model Performance:**

- Strong performance for stromal cells (F1: 0.86) and immune cells (F1: 0.84)
- Good performance for white adipocytes (F1: 0.72)
- Moderate performance for beige adipocytes (F1: 0.63)
- Poor performance for brown adipocytes (F1: 0.00) due to limited training data



### Visualization
I have implemented visualization features to evaluate model performance:

- Confusion matrix showing the classification accuracy for each cell type
- Feature importance analysis highlighting the most informative features
- t-SNE plots of cellular data with predicted cell type annotations

### Cell Type Annotator
The current version of my annotator can:

- Read single-cell RNA-seq data in standard formats
- Preprocess the data using standard bioinformatic approaches
- Identify clusters of cells using K-means clustering
- Predict the most likely cell type for each cluster based on marker gene expression
- Provide confidence scores for predictions
  
## **Goals**
The goal of this project is to create an automated adipocyte cell annotator that identifies ideal cluster resolution and predicts most-likely cell types of clusters. Along with predicting celltypes based on known markers, this annotator will identify unknown or new celltypes for user review to help aid in the identification of potential new cell types and/or pathways.

## **Next Steps**
1. **Data Expansion:**
   - Add datasets with better representation of brown adipocytes
   - Include cold-stimulated datasets for better beige adipocyte identification
   - Add obesity-related datasets to improve disease state recognition
2. **Model Refinement:**
   - Implement hierarchical clustering for multi-level cell type classification
   - Add novel cell type identification based on confidence scores
   - Improve feature engineering to better capture cell state transitions
3. **Visualization Enhancement:**
   - Develop hierarchical cell marker tree visualization
   - Add interactive visualizations for exploring clustering results
   - mplement confidence visualization for potentially novel cell types

## **Data Collection**
Currently utilizing public data sets from the National Library of Medicine - National Center for Biotechnology Information (NLM-NCBI) genomics data repository, the Gene Expression Omnibus (GEO). I have processed two key datasets (GSE272938 - adipocyte de-differentiation study and GSE266326 - influenza infection study). I plan to expand to additional datasets to ensure comprehensive coverage of adipocyte types (white, brown, and beige), species (human and mice), and adipocyte cell subtypes.

## **Data Visualization**
The current implementation includes t-SNE visualization of clustering and cell type predictions. I am developing a hierarchical cell marker tree that will better represent relationships between cell types at different clustering resolutions. This visualization will include confidence scores for predictions and highlight potentially novel cell types or states.

## **Test Plan**
I am using a 70-20-10 split for training, validation, and test data. The current model has been evaluated using 5-fold cross-validation (mean score: 0.76) and testing on a held-out portion of the data (accuracy: 83%). As more datasets are added, I will maintain this rigorous validation approach, ensuring that the model generalizes well to unseen data from different experimental conditions, tissues, and species.
