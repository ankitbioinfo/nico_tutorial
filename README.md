# NiCo Tutorial

<div align="center">

<img src="Figure1old.png" width="640px" />

</div>

## Table of Contents

-   [NiCo](#nico)
    -   [Table of Contents](#table-of-contents)
    -   [Overview](#overview)
        -   [About Tutorial](#background)
    -   [Getting Started](#getting-started)
        -   [Prerequisites](#prerequisites)
        -   [Installation](#installation)
        -   [Data preparation](#preparation-NiCo)
        -   [Running NiCo](#running-NiCo)
    -   [Documentation](#documentation)
    -   [MERSCOPE data](#Vizgen-MERSCOPE-data)
    -   [Citing NiCo](#citing-nico)
    -   [Contact](#contact)

## Overview

We have developed the NiCo (Niche Covariation) package for the integration of single-cell resolution spatial transcriptomics and scRNA-seq data to (1) perform cell type annotations in the spatial modality by label transfer, (2) predict niche cell type interactions within local neighborhoods, and (3) infer cell state covariation and the underlying molecular crosstalk in the niche. NiCo infers factors capturing cell state variability in both modalities and identifies genes correlated to these latent factors for the prediction of ligand-receptor interactions and factor-associated pathways.

### About Tutorial
NiCo can run any spatial technolgoies but it is mainly designed for single-cell resolution of spatial technologies such as MERFISH/SEQFISH/XENIUM etc. 
We are providing a first tutorial on running the NiCo pipeline for the data integration of single-cell RNA sequencing (reference) and single-cell resolution of spatial transcriptomics data (query). This tutorial explains all steps of the NiCo pipeline, i.e., annotation of cell types in the spatial modality by label transfer from the scRNA-seq data, prediction of significant niche interactions, and derivation of cell state covariation within the local niche. The ###second### NiCo tutorial is made for low resoution sequencing based technologies where integration with single-cell RNA sequencing is not required.  

Please keep all the files (NiCoLRdb.txt and *.ipynb) and folders (inputRef, inputQuery) in the same path to complete the tutorial. 

### Prerequisites

Please follow the information provided at [nico-sc-sp pip repository](https://pypi.org/project/nico-sc-sp/)


### Installation
For detailed instruction please follow the instructions at [nico-sc-sp pip repository](https://pypi.org/project/nico-sc-sp/)

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


### Tuturial 1A: Data preparation for high resolution spatial technologies 

For data preparation, first extract all zip files and run the Juypter notebook Start_Data_preparation_for_niche_analysis.ipynb.
The newer version of Start_Data_prep_new.ipynb is available for updated version. 

### Tutorial 1B: Running NiCo for high resolution spatial technologies 
After running this script and having generated normalised data files, run the Jupyter notebook Perform_spatial_analysis.ipynb (Perform_spatial_analysis_new.ipynb for newer version) to perform the core steps of NiCo.

By default, the tutorial generates all the figures both in the respective directory and inside the notebook. Please refer to the documentation for details on functions and parameters. 

### Tutorial 2: Running NiCo for sequencing based spatial technologies 
To run the NiCo on cerebellum data from Slide-seqV2 technology [data is taken from figure 3 of Cable, D. M. et al. Nature methods 19, 1076–1087 (2022)]. <br> 
The NiCo niche detection and covariation analysis task is shown in following [jupyter notebook nico_analysis_lowres_seq_tech](nico_analysis_lowres_seq_tech.ipynb)
Please download the data from [this link](https://www.dropbox.com/scl/fi/6hxyp2pxpxalw9rfirby6/nico_cerebellum.zip?rlkey=9ye6rsk92uj9648ogjw5ypcum&st=lvc8e366&dl=0)
and keep the data in following path ``nico_cerebellum/cerebellum.h5ad`` to complete the tutorial.  

### Warnings 
If at any step it shows the warning ```findfont: Font family 'Helvetica' not found```
Then please initialize the matplotlibrc file to use different font as 

```
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans','Lucida Grande', 'Verdana']
```


## Documentation 

The detailed [documentation of NiCo](https://nico-sc-sp.readthedocs.io/en/latest/) modules and their usage functions can be seen in the given link. 

## Vizgen MERSCOPE data 
If you are working with Vizgen MERSCOPE spatial data, please process with "process_vizgenData.py" script to convert Vizgen data into gene_by_cell.csv and tissue_positions_list.csv files. 

## Cellbarcode name 
If you encounter any issues while running, please ensure that the cell barcode name is composed of characters rather than integer numbers. When pandas reads numeric values, it will read as int64 instead of object which create datatype confusion for other parts of the code. Therefore, please convert your cell barcode numbers to strings if they are purely numeric.

## Citing NiCo

-   Ankit Agrawal, Stefan Thomann, Dominic Grün. NiCo Identifies Extrinsic Drivers of Cell State Modulation by Niche Covariation Analysis.
    ***Submitted***, 2024

## Contact

> **_contact:_** If you face any problem during the tutorial or have any questions, please email me (ankitplusplus at gmail.com) or raise an issue on Git Hub. 


## License 
[MIT License](LICENSE)

