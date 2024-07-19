# NiCo Package

<div align="center">

<img src="Figure1old.png" width="640px" />

</div>

# Table of Contents

-   [NiCo](#nico)
    -   [Table of Contents](#table-of-contents)
    -   [Overview](#overview)
    -   [Getting Started](#getting-started)
        -   [About Tutorial](#background)
        -   [Prerequisites](#prerequisites)
        -   [Installation](#installation)
    -   [Notes and Warnings](#Font)
    -   [Documentation](#documentation)
    -   [Citing NiCo](#citing-nico)
    -   [Contact](#contact)

# Overview

We have developed the NiCo (Niche Covariation) package for the integration of single-cell resolution spatial transcriptomics and scRNA-seq data. NiCo allows you to:

(1) Perform cell type annotations in the spatial modality by label transfer. <br>
(2) Predict niche cell type interactions within local neighborhoods.  <br>
(3) Infer cell state covariation and the underlying molecular crosstalk in the niche. 

NiCo infers factors capturing cell state variability in both modalities and identifies genes correlated to these latent factors to predict ligand-receptor interactions and factor-associated pathways.

# Tutorial
NiCo can run any spatial technologies, but it is mainly designed for single-cell resolution spatial technologies such as MERFISH, SEQFISH, and XENIUM. 

We are providing a **first tutorial** on running the NiCo pipeline for the data integration of single-cell RNA sequencing (reference) and single-cell resolution of spatial transcriptomics data (query). 

The **second NiCo tutorial** is designed for low-resoution sequencing-based spatial transcriptomics technologies where integration with single-cell RNA sequencing is not required. 

Please get the tutorial link below and keep all the files (NiCoLRdb.txt and *.ipynb) and folders (inputRef, inputQuery) in the same path to complete the tutorial. 


## Tuturial 1: High-resolution spatial technologies

* **Part A: Data Preparation** 
Extract all zip files and run the Juypter notebook [``Start_Data_prep_new.ipynb``](Start_Data_prep_new.ipynb) to create the data files for NiCo analysis. 

* **Part B: Running NiCo on selected cell types**
After data preparation, generating normalised data files, run the Jupyter notebook [``nico_analysis_highres_image_tech.ipynb``](nico_analysis_highres_image_tech.ipynb) to perform the core steps of NiCo. <br>
This tutorial explains all steps of the NiCo pipeline, including annotation of cell types in the spatial modality by label transfer from the scRNA-seq data, prediction of significant cell type niche interactions, and derivation of cell state covariation within the local niche.  <br>
By default, the figures generated are saved both in the respective directory and inside the notebook. <br>
Please refer to the documentation for details on functions and parameters. <br>
The data source is provided in the manuscript. 

* **Part B: Running NiCo on all cell types**
If users want to perform NiCo analysis on the full data without specifying any cell type, refer to the script. [``nico_analysis_highres_image_tech.py``](nico_analysis_highres_image_tech.py). The output log can be seen [here](log_output.txt). Due to large number of images, Jupyter notebook might not display them properly. However, leaving the ``choose_celltypes`` and ``choose_factors_id`` lists blank will enable the full analysis.

## Tutorial 2: Sequencing-based spatial technologies 
To run NiCo on cerebellum data from Slide-seqV2 technology [data from Cable, D. M. et al. Nature methods 19, 1076–1087 (2022)]. <br> 
* Download the data from [this link](https://www.dropbox.com/scl/fi/6hxyp2pxpxalw9rfirby6/nico_cerebellum.zip?rlkey=9ye6rsk92uj9648ogjw5ypcum&st=lvc8e366&dl=0)
and place the data in following path ``nico_cerebellum/cerebellum.h5ad`` to complete the tutorial.  
* The NiCo niche detection and covariation analysis task can be run via following jupyter notebook [``nico_analysis_lowres_seq_tech.ipynb``](nico_analysis_lowres_seq_tech.ipynb)



# Prerequisites

Please follow the instructions provided in the [nico-sc-sp pip repository](https://pypi.org/project/nico-sc-sp/) for set up and installation. 


# Installation
For detailed instruction,  visit the [nico-sc-sp pip repository](https://pypi.org/project/nico-sc-sp/)

``` console
conda create -n nicoUser python=3.11
conda activate nicoUser
pip install nico-sc-sp
```
Sometimes, the pygraphviz package cannot be installed via pip, or during the cell type interaction part it gives error that "neato" not found in path so an alternative conda way of installation is recommended. Please follow the installation of pygraphviz [here](https://pygraphviz.github.io/documentation/stable/install.html)


``` console
conda create -y -n nicoUser python=3.11
conda activate nicoUser
conda install -c conda-forge pygraphviz
pip install nico-sc-sp
pip install jupyterlab
```




# Notes and Warnings 
### Font 
If at any step it shows the warning ```findfont: Font family 'Helvetica' not found```
Then please initialize the matplotlibrc file to use different font as 

```
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans','Lucida Grande', 'Verdana']
```

### Vizgen MERSCOPE data 
If you are working with Vizgen MERSCOPE spatial data, please process with "process_vizgenData.py" script to convert Vizgen data into gene_by_cell.csv and tissue_positions_list.csv files. 

### Cellbarcode name 
If you encounter any issues while running, please ensure that the cell barcode name is composed of characters rather than integer numbers. When pandas reads numeric values, it will read as int64 instead of object which create datatype confusion for other parts of the code. Therefore, please convert your cell barcode numbers to strings if they are purely numeric.


# Documentation 

The detailed documentation on NiCo modules and their usage, visit [NiCo documentation](https://nico-sc-sp.readthedocs.io/en/latest/).


# Citing NiCo
If you use NiCo in your research, please cite it as follows: 

-   Ankit Agrawal, Stefan Thomann, Sukanya Basu, Dominic Grün. NiCo Identifies Extrinsic Drivers of Cell State Modulation by Niche Covariation Analysis.
    ***Submitted***, 2024

# Contact

> **_contact:_** If you face any problem during the tutorial or have any questions, please email me (ankitplusplus at gmail.com) or raise an issue on Git Hub. 


## License 
[MIT License](LICENSE)

