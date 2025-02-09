# **Automated Adipocyte Cell Identifier and Annotator** #

## Description
For the past year, I have worked under Dr. Nabil Rabhi in the Biochemistry Department over at Medical Campus. There I have learned the basics of the single-cell RNA sequencing pipeline, doing quality control, annotation, and now analysis. Through this process I found there are several rather tedious steps in the process that have the potential to be automated with specialized packages and models. There exists several annotation packages in which you can train the model with self-provided data sets. However, very few packages exist that are pre-trained. My proposed package would be one such pre-trained model, specialized on adipose tissue. This model would then be able to be used in my, as well as others, analysis of adipose tissue, allowing for an ease of identification and annotation of sequenced data.
### Cell Type Annotator
My proposed annotator would be a model trained with existing datasets. The model will be able to take an input data set, identify ideal clustering resolution(s) [see Heirarchical Annotation subheading], identify most likely cell type by cluster based on existing knowledge of cell markers, and identify potential new cell types or pathways.
#### Heirarchical Annotation
### Identification of Unknown/New cell types and pathways
### Existing Resources
While this annotator may seem ambitious, my year of experience has lent me some invaluable resources. In preparing to process the current dataset, we (the lab) created an extensive adipocyte cell marker database from existing published works. This will reduce the work time for data collection and referencing as I already have access to known markers, allowing me to dedicate more time to the contruction and refinement of my model. While our cell marker database will eliminate some required reference data gathering, I plan to use that additional time to establish an extensive dataset with which to train my model [see Data Collection subheading].

## Goals
The goal of this project is to create an automated adipocyte cell annotator that identifies ideal cluster resolution and annotates clusters. Along with assigning celltypes based on known markers, this annotator will identify unknown or new celltypes for user review to help aid in the identification of potential new cell types and/or pathways.

## Data Collection
I plan to a utilize public data sets from the National Library of Medicine - National Center for Biotechnology Information (NLM-NCBI) genomics data repository, the Gene Expression Omnibus (GEO) [https://www.ncbi.nlm.nih.gov/gds/?term=adipose]. There are over 75,000 public datasets containing adipose tissue sequencing. I will then select a subset of these to use as my testing/training sets. While I do not have a specific benchmark for the number of datasets I plan to include in my project dataset, I want to ensure thorough coverage of adipocyte types (white, brown, and beige), species (human and mice), and adipocyte cell suptypes; a complete dataset for this project will include multiple examples of both species and all adipocyte types and subtypes.

## Data Visualization
The overall clustering and type annotation will be visualized with a t-SNE plot, however a 

## Test Plan
