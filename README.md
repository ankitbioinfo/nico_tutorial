# NiCo Tutorial

We are providing a tutorial on running the NiCo pipeline for the data integration of single-cell RNA sequencing (reference) and single-cell resolution of spatial transcriptomics data (query). This tutorial explains all steps of the NiCo pipeline, i.e., annotation of cell types in the spatial modality by label transfer from the scRNA-seq data, prediction of significant niche interactions, and derivation of cell state covariation within the local niche. 

For data preparation, first extract all zip files and run the Juypter notebook Start_Data_preparation_for_niche_analysis.ipynb.

After running this script and having generated normalised data files, run the Jupyter notebook Perform_spatial_analysis.ipynb to perform the core steps of NiCo.

By default, the tutorial generates all the figures both in the respective directory and inside the notebook. Please refer to the documentation for details on functions and parameters. 

The detailed documentation of NiCo modules and their functions can be seen here. 

https://nico-sc-sp.readthedocs.io/en/latest/
