import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import os
import sys
#import pickle
import math
import collections
import networkx as nx
from types import SimpleNamespace
from scipy.spatial import cKDTree
import scipy.sparse as scipy_sparse

import warnings
import time
import seaborn as snn
from collections import Counter


fpath=os.path.join(os.path.dirname(__file__),'utils')
sys.path.append(fpath)
from SCTransform import SCTransform

#warnings.filterwarnings('ignore')
#export PYTHONWARNINGS='ignore:Multiprocessing-backed parallel loops:UserWarning'
#os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


def create_directory(outputFolder):
    """
    Create an empty directory.

    This function checks if a specified directory exists, and if not, it creates the directory.

    Parameters
    ----------
    outputFolder : str
        The path of the directory to be created.

    Raises
    ------
    OSError
        If the directory cannot be created due to permission issues or other OS-related errors.

    Notes
    -----
    - If the directory already exists, no action is taken.
    - This function ensures that the directory path is available for subsequent file operations.

    Example
    -------
    >>> create_directory('./new_out/')
    """
    answer=os.path.isdir(outputFolder)
    if answer==True:
        pass
    else:
        os.mkdir(outputFolder)


def find_index(sp_genename,sc_genename):
    """
    Find the common gene space submatrix between two modalities.

    This helper function is used within the `find_anchor_cells_between_ref_and_query` function to identify the indices of common genes
    between two lists of gene names corresponding to spatial and scRNAseq modalities.

    Parameters
    ----------
    sp_genename : list
        A list of gene names from the spatial modality.

    sc_genename : list
        A list of gene names from the scRNAseq modality.

    Returns
    -------
    list
        A list of indices corresponding to the common genes found in both `sp_genename` and `sc_genename`.

    Example
    -------
    >>> sp_genes = ['gene1', 'gene2', 'gene3', 'gene4']
    >>> sc_genes = ['gene3', 'gene4', 'gene5', 'gene6']
    >>> index_sp,index_sc = find_index(sp_genes, sc_genes)
    >>> print(index_sp)
    [2, 3]
    """

    index_sc=[]
    index_sp=[]
    d={}
    for j in range(len(sc_genename)):
        name=sc_genename[j]
        d[name]=j

    for i in range(len(sp_genename)):
        name=sp_genename[i]
        try:
            d[name]
            flag=1
        except KeyError:
            flag=0
        if flag==1:
            index_sc.append(d[name])
            index_sp.append(i)
    return index_sp,index_sc


def find_match_index_in_dist(t1,t2,s1,s2,index_1,index_2):
    """
    Helper function used in `find_mutual_nn` to find the correct pairing of cell barcodes.

    This function takes two lists of cell barcodes and their corresponding indices and returns the matched pair of distances for the specified indices.

    Parameters
    ----------
    t1 : list or numpy.ndarray
        A list or array containing the distances corresponding to the cell barcodes in `s1`.
    t2 : list or numpy.ndarray
        A list or array containing the distances corresponding to the cell barcodes in `s2`.
    s1 : list or numpy.ndarray
        A list or array containing the cell barcodes for the first set of distances.
    s2 : list or numpy.ndarray
        A list or array containing the cell barcodes for the second set of distances.
    index_1 : int
        The index of the cell barcode in `s1` to be matched.
    index_2 : int
        The index of the cell barcode in `s2` to be matched.

    Returns
    -------
    tuple
        A tuple (p1, p2) where p1 is the distance from `t1` corresponding to `index_1` and p2 is the distance from `t2` corresponding to `index_2`.

    """
    for i in range(len(s1)):
        if s1[i]==index_1:
            p1=t1[i]
    for i in range(len(s2)):
        if s2[i]==index_2:
            p2=t2[i]
    return p1,p2


def find_mutual_nn(minkowski_order,data1, data2, sp_barcode,sc_barcode, k1, k2):
    """
    Helper function used in `find_anchor_cells_between_ref_and_query` to find mutual nearest neighbors using cKDTree.

    This function finds mutual nearest neighbors (MNNs) between two datasets using the cKDTree algorithm. The mutual nearest neighbors are those pairs of points that are each other's nearest neighbors.

    Parameters
    ----------
    minkowski_order : int
        The order of the Minkowski distance to use. For example, 2 is the Euclidean distance.
    data1 : numpy.ndarray
        The reference dataset, typically the spatial dataset.
    data2 : numpy.ndarray
        The query dataset, typically the sequencing dataset.
    sp_barcode : numpy.ndarray
        Array of barcodes for the spatial dataset.
    sc_barcode : numpy.ndarray
        Array of barcodes for the single-cell dataset.
    k1 : int
        The number of nearest neighbors to query in the spatial dataset.
    k2 : int
        The number of nearest neighbors to query in the single-cell dataset.

    Returns
    -------
    numpy.ndarray
        An array where each row represents a mutual nearest neighbor pair and their distances. The columns are:
        [sp_barcode of mutual_1, sc_barcode of mutual_2, distance in data1, distance in data2]

    """
    #data1 is spatial
    #data2 is single

    n_jobs=-1
    d1,k_index_1 = cKDTree(data1).query(x=data2, k=k1, p=minkowski_order,workers=n_jobs)
    d2,k_index_2 = cKDTree(data2).query(x=data1, k=k2, p=minkowski_order,workers=n_jobs)
    #print(data1.shape,k_index_1.shape,'\t',data2.shape, k_index_2.shape)
    mutual_1 = []
    mutual_2 = []
    dist_1=[]
    dist_2=[]
    for index_2 in range(data2.shape[0]):
     t1=d1[index_2]
     s1=k_index_1[index_2]
     for index_1 in s1:
        t2=d2[index_1]
        s2=k_index_2[index_1]
        if index_2 in s2:
            p1,p2=find_match_index_in_dist(t1,t2,s1,s2,index_1,index_2)
            mutual_1.append(index_1)
            mutual_2.append(index_2)
            dist_1.append(p1)
            dist_2.append(p2)

    a1=np.array(mutual_1)
    a2=np.array(mutual_2)
    dist_1=np.array(dist_1)
    dist_2=np.array(dist_2)
    a1=sp_barcode[a1]
    a2=sc_barcode[a2]
    a1=np.reshape(a1,(1,len(a1)))
    a2=np.reshape(a2,(1,len(a2)))
    dist_1=np.reshape(dist_1,(1,len(dist_1)))
    dist_2=np.reshape(dist_2,(1,len(dist_2)))
    b=np.concatenate((a1,a2,dist_1,dist_2)).T

    return b




def sct_return_sc_sp_in_shared_common_PC_space(ad_sp1,ad_sc1,no_of_pc,method):
    """
    Helper function used in `find_anchor_cells_between_ref_and_query` to transform the common gene expression data into PCA space.

    This function scales the data, performs PCA on single-cell data, and then projects both spatial and single-cell data into the shared PCA space. The projections are normalized by z-scores.

    Parameters
    ----------
    ad_sp1 : AnnData
        AnnData object containing the spatial gene expression data.
    ad_sc1 : AnnData
        AnnData object containing the sequencing gene expression data.
    no_of_pc : int
        Number of principal components to compute.
    method : str
        Method used for scaling and PCA.

    Returns
    -------
    transfer_sp_com : numpy.ndarray
        The spatial data projected into the shared PCA space.
    transfer_sc_com : numpy.ndarray
        The single-cell data projected into the shared PCA space.
    sp_barcode : numpy.ndarray
        Barcodes of the spatial data.
    sc_barcode : numpy.ndarray
        Barcodes of the single-cell data.

    """

    sct_ad_sc=ad_sc1.copy()
    sct_ad_sp=ad_sp1.copy()

    sc.pp.scale(sct_ad_sp, zero_center=True)
    sc.pp.scale(sct_ad_sc, zero_center=True)

    sc.pp.pca(sct_ad_sc,zero_center=None,n_comps=no_of_pc)
    sc_com_pc=sct_ad_sc.varm['PCs']

    if scipy_sparse.issparse(sct_ad_sc.X):
       msc=sct_ad_sc.X.toarray()
    else:
       msc=sct_ad_sc.X

    if scipy_sparse.issparse(sct_ad_sp.X):
       msp=sct_ad_sp.X.toarray()
    else:
       msp=sct_ad_sp.X

    msp=np.nan_to_num(msp)
    msc=np.nan_to_num(msc)
    transfer_sp_com = np.matmul(msp, sc_com_pc)
    transfer_sc_com = np.matmul(msc, sc_com_pc)

    for i in range(transfer_sp_com.shape[1]):
        mu1=np.mean(transfer_sp_com[:,i])
        svd1=np.std(transfer_sp_com[:,i])
        transfer_sp_com[:,i]= (transfer_sp_com[:,i]-mu1)/svd1

        mu2=np.mean(transfer_sc_com[:,i])
        svd2=np.std(transfer_sc_com[:,i])
        transfer_sc_com[:,i]= (transfer_sc_com[:,i]-mu2)/svd2
        #print(i,mu1,mu2,svd1,svd2)



    sc_barcode=sct_ad_sc.obs_names.to_numpy()
    sp_barcode=sct_ad_sp.obs_names.to_numpy()

    #print('sc',transfer_sc_com.shape,sc_cellname.shape)
    #print('sp',transfer_sp_com.shape,sp_cellname.shape)

    return transfer_sp_com, transfer_sc_com, sp_barcode,sc_barcode





def find_annotation_index(annot_cellname,sct_cellname):
    """
    Helper function for `find_common_MNN` to find the correct cell name.

    This function maps the indices of cell barcodes names to the corresponding indices in the anchored data.

    Parameters
    ----------
    annot_cellname : list or array-like
        List or array of cell barcode names.
    sct_cellname : list or array-like
        List or array of anchored cell barcode names.

    Returns
    -------
    index : list
        List of indices in `sct_cellname` that corresponds to the position of the cell barcode name in `annot_cellname`.

    """
    d={}
    for i in range(len(annot_cellname)):
        d[annot_cellname[i]]=i

    index=[]
    for i in range(len(sct_cellname)):
        index.append(d[sct_cellname[i]])

    return index




def find_commnon_MNN(input):
    """
    Helper function used in `find_anchor_cells_between_ref_and_query` to find the anchored cells between
    spatial and sequencing modalities using the mutual nearest neighbors (MNN) method in the PCA space.

    This function reads MNN pairs from a provided file and processes the data to identify mutual nearest neighbors
    between spatial and single-cell datasets.

    Parameters
    ----------
    input : object
        An object containing various attributes required for the function.

    Returns
    -------
    data : numpy.ndarray
        After prunning the bad anchors it returen the good anchors.

    """
    #df=pd.read_csv(input.fname_mnn_anchors,header=None)
    #data contains 2 column files of sct_pairing_shared_common_gene_PC.csv
    # first column is MNN pairs of spatial and
    # second column is MNN pairs of single cell
    #data=df.to_numpy()
    data=input.fname_mnn_anchors

    mnn_singlecell_matchpair_barcode_id=np.unique(data[:,1])
    mnn_spatial_matchpair_barcode_id=np.unique(data[:,0])

    # find the annotated indexes
    index_annot_sc=find_annotation_index(input.annotation_singlecell_barcode_id,mnn_singlecell_matchpair_barcode_id)
    index_annot_sp=find_annotation_index(input.annotation_spatial_barcode_id,mnn_spatial_matchpair_barcode_id )



    #There are many indexes for spatial and single cell data
    # 1) MNN single cell                    data[:,1]                                       90,876
    # 2) MNN unique                          mnn_singlecell_matchpair_id                    10,089
    # 3) SC transform cell id                input.sct_singlecell_barcode_id                18,754
    # 4) original matrix cell id             input.annotation_singlecell_barcode_id         185,894
    # 5) original cell type name            input.annotation_singlecell_celltypename        185,894
    # 6) MNN unique id in sct               mnn_singlecell_matchpair_barcode_id             10,089
    # 7) common index between 6 and 4       index_mnn_sc,index_annot_sc

    # 1) MNN spatial                        data[:,0]                                       90,876
    # 2) MNN unique                         mnn_spatial_matchpair_id                        8,932
    # 3) SC transform cell id               input.sct_spatial_barcode_id                    86,880
    # 4) original matrix cell id            input.annotation_spatial_barcode_id             395,215
    # 5) original cell type name            input.annotation_spatial_celltypename           395,215
    # 55) original spatial cluster id       input.annotation_spatial_cluster_id             395,215
    # 6) MNN unique id in sct               mnn_spatial_matchpair_barcode_id                8,932
    # 7) common index between 6 and 4       index_mnn_sp,index_annot_sp

    d_single_cluster={}
    for i in range(len(input.lsc[0])):
        singlecell_unique_clusterid=input.lsc[1][i]
        d_single_cluster[singlecell_unique_clusterid]=i

    d_spatial_cluster={}
    for i in range(len(input.lsp[0])):
        spatialcell_unique_clusterid=input.lsp[1][i]
        d_spatial_cluster[spatialcell_unique_clusterid]=i

    total_in_row=np.zeros((1,len(input.lsp[0])),dtype=float)
    total_in_col=np.zeros((1,len(input.lsc[0])),dtype=float)

    d_single={}
    for i in range(len(input.annotation_singlecell_cluster_id)):
        d_single[input.annotation_singlecell_barcode_id[i]]=input.annotation_singlecell_cluster_id[i]
        col=d_single_cluster[d_single[input.annotation_singlecell_barcode_id[i]]]
        total_in_col[0,col]+=1

    d_spatial={}
    for i in range(len(input.annotation_spatial_cluster_id)):
        d_spatial[input.annotation_spatial_barcode_id[i]]=input.annotation_spatial_cluster_id[i]
        spatialcell_cluid=d_spatial[input.annotation_spatial_barcode_id[i]]
        col=d_spatial_cluster[spatialcell_cluid]
        total_in_row[0,col]+=1


    mat21=np.zeros((len(input.lsc[0]),len(input.lsp[0])),dtype=float)
    mat22=np.zeros((len(input.lsp[0]),len(input.lsc[0])),dtype=float)
    mat1=np.zeros(  (1,len(input.lsc[0]) ) ,dtype=float)
    mat3=np.zeros(  (1,len(input.lsp[0]) ),dtype=float)


    unique_singlecell_barcode_in_MNN=np.unique(data[:,1])
    for i in range(len(unique_singlecell_barcode_in_MNN)):
        singlecell_cluid=d_single[unique_singlecell_barcode_in_MNN[i]]
        col=d_single_cluster[singlecell_cluid]
        #print(i,spatialcell_cluid)
        mat1[0,col]+=1


    #count how many anchor points matches to each spatial clusters
    unique_spatial_barcode_in_MNN=np.unique(data[:,0])
    for i in range(len(unique_spatial_barcode_in_MNN)):
        spatialcell_cluid=d_spatial[unique_spatial_barcode_in_MNN[i]]
        col=d_spatial_cluster[spatialcell_cluid]
        mat3[0,col]+=1


    anchorFreqRow=mat3/total_in_row
    anchorFreqCol=mat1/total_in_col

    save_anchors={}
    for i in range(len(data)):
            spatialcell_cluid=d_spatial[data[i,0]]
            singlecell_cluid=d_single[data[i,1]]
            col=d_spatial_cluster[spatialcell_cluid]
            row=d_single_cluster[singlecell_cluid]
            mat21[row,col]+=1
            mat22[col,row]+=1
            key=str(col)+'#'+str(row)
            name=data[i,0]+'#'+data[i,1]
            if key not in save_anchors:
                save_anchors[key]=[name]
            else:
                if name not in save_anchors[key]:
                    save_anchors[key].append(name)

    #col normalization
    for i in range(len(mat21[0])):
        mat21[:,i]=mat21[:,i]/np.sum(mat21[:,i])

    for i in range(len(mat22[0])):
        mat22[:,i]=mat22[:,i]/np.sum(mat22[:,i])


    newmat2=np.vstack((anchorFreqCol,mat22))
    mat2=np.vstack((anchorFreqRow,mat21))
    cname2=input.lsp[0]
    newcname2=input.lsc[0]


    #fw=open(input.savepath+"spatial_annotation_along_SP.dat",'w')
    unique_rep_of_leiden_clusters_in_sp={}
    for i in range(mat2.shape[1]):
        af=mat2[0,i]
        col=mat2[1:,i]
        index=np.argsort(-col)
        found=''
        for j in range(len(col)):
            value=col[index[j]]
            nct=input.lsc[0][index[j]]
            if value>input.across_spatial_clusters_dispersion_cutoff:
                if j!=0:
                    found+=', '
                found+=nct+':'+'%0.3f'%value
                if cname2[i] not in unique_rep_of_leiden_clusters_in_sp:
                    unique_rep_of_leiden_clusters_in_sp[cname2[i]]=[nct]
                else:
                    if nct not in unique_rep_of_leiden_clusters_in_sp[cname2[i]]:
                        unique_rep_of_leiden_clusters_in_sp[cname2[i]].append(nct)
        #fw.write(str(i)+'\t'+cname2[i]+'\tF='+str('%0.3f'%af)+',\t'+found+'\n')

    #these clusters should not be removed
    low_anchors_spatial_clusters={}
    for key in unique_rep_of_leiden_clusters_in_sp:
        temp=unique_rep_of_leiden_clusters_in_sp[key]
        if len(temp)==1:
            low_anc_ct=temp[0]
            if low_anc_ct not in low_anchors_spatial_clusters:
                low_anchors_spatial_clusters[low_anc_ct]=[key]
            else:
                low_anchors_spatial_clusters[low_anc_ct].append(key)

    #print("low anchors",low_anchors_spatial_clusters)
    #{'KCs': ['c11', 'c7'], 'Stellatecells': ['c12'], 'Cholangiocytes': ['c16'], 'Bcells': ['c17'], 'LSECs': ['c18', 'c6']}


    #fw.close()
    #fw=open(input.savepath+"spatial_annotation_along_SC.dat",'w')
    good_anchors={}
    tt=[]
    for i in range(newmat2.shape[1]):
        af=newmat2[0,i]
        col=newmat2[1:,i]
        index=np.argsort(-col)
        found=''
        for j in range(len(col)):
            flag=0
            if col[index[j]]>input.across_spatial_clusters_dispersion_cutoff:
                flag=1
                # this flag is true if spillovered anchores belong to other leiden cluster are > dispersion cutoff
            elif newcname2[i] in low_anchors_spatial_clusters:
                if input.lsp[0][index[j]] in low_anchors_spatial_clusters[newcname2[i]]:
                    flag=0
                # this flag is true if spillovered anchores belong to other leiden cluster < dispersion but uniquly mapped

            if flag==1:
                if j!=0:
                    found+=', '
                found+=input.lsp[0][index[j]]+':'+'%0.3f'%col[index[j]]

                key=str(index[j])+'#'+str(i)
                tt.append(key)
                if key in save_anchors:
                    list_of_anchors=save_anchors[key]
                    for k in range(len(list_of_anchors)):
                        name=list_of_anchors[k]
                        #print(name)
                        if name not in good_anchors:
                            good_anchors[name]=1
                        else:
                            good_anchors[name]+=1

        #fw.write(str(i)+'\t'+newcname2[i]+'\tF='+str('%0.3f'%af)+',\t'+found+'\n')
    #fw.close()

    c=0
    for key in good_anchors:
        c+=good_anchors[key]

    count=0
    ca={}
    for key in save_anchors:
        list_of_anchors=save_anchors[key]
        count+=len(list_of_anchors)
        for j in range(len(list_of_anchors)):
            ca[list_of_anchors[j]]=1

    colname=['total # of sc', 'total # of sp']
    cname1=['anchorFreq']+list(input.lsc[0])
    visualize=[mat2,cname1,cname2]
    return good_anchors,visualize


def visualize_spatial_anchored_cell_mapped_to_scRNAseq(input,saveas='pdf',transparent_mode=False,showit=True,figsize=(12,10)):
    """
    Visualizes the anchored cells mapping between spatial and sequencing modalities.

    This function generates a heatmap to visualize the mapping of anchored cells between spatial Leiden clusters and scRNAseq clusters.

    Parameters
    ----------
    input : object,
        An object containing various attributes required for the function. Specifically, it must contain:
        - `visualize_anchors` : tuple
        A tuple containing the matrix of anchored cells, and the cluster names for spatial and scRNAseq data.
        Example: (matrix, spatial_cluster_names, scrnaseq_cluster_names)
        - `KNN` : int
        The number of nearest neighbors used in the mutual nearest neighbors (MNN) analysis.
        - `output_annot` : str
        The path where the output figure will be saved.
    saveas : str, optional
        The format to save the figure, either 'pdf' or 'png' (default is 'pdf').
    transparent_mode : bool, optional
        Whether the background of the figure should be transparent (default is False).
    showit : bool, optional
        Whether to display the figure immediately (default is True). If False, the figure is closed after saving.
    figsize : tuple, optional
        The size of the figure (default is (12,10)).

    Outputs
    -------
    The figure is saved at the location specified by "nico_out/annotations/".
    """

    mat2,cname1,cname2=input.visualize_anchors
    fig=plt.subplots(1,1,figsize=figsize)
    snn.heatmap(data=mat2,annot=True, fmt='0.2f',xticklabels=cname2, annot_kws={"size": 5},yticklabels=cname1)
    plt.xlabel('Spatial Leiden Clusters')
    plt.ylabel('scRNAseq Clusters')
    plt.title('MNN K = ' + str(input.KNN),fontsize=12)
    plt.tight_layout()
    print("The figures are saved: ", input.output_annot+'visualize_anchors.'+saveas)
    plt.savefig(input.output_annot+'visualize_anchors.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
    if showit:
        pass
    else:
        plt.close('all')





def find_anchor_cells_between_ref_and_query(refpath='./inputRef/',quepath='./inputQuery/',
output_annotation_dir=None,
output_nico_dir=None,
neigh=50,no_of_pc=50,minkowski_order=2):

    """
    Finds all anchor cells between query and reference data.

    This function reads the reference and query data from the specified directories, performs necessary preprocessing, and finds mutual nearest neighbors in the PCA space to map cell types between the two datasets.


    Parameters
    ----------
    refpath : str, optional
        Path to the directory containing reference scRNAseq data. This directory should contain:
        - 'Original_counts.h5ad' : The original count matrix in raw layer.
        - 'sct_singleCell.h5ad' : The scTransform-like normalized matrix.
        (default is './inputRef/')
    quepath : str, optional
        Path to the directory containing spatial transcriptomics query data. This directory should contain:
        - 'sct_spatial.h5ad' : The scTransform-like normalized matrix.
        (default is './inputQuery/')
    output_annotation_dir : str, optional
        Directory to save output annotations. If None, a default directory is used.
        (default is None)
    output_nico_dir : str, optional
        Directory to save output NICOLAE results. If None, './nico_out/' is used.
        (default is './nico_out/annotations')
    neigh : int, optional
        The number of K-nearest neighbors to find the anchor cells.
        (default is 50)
    no_of_pc : int, optional
        The number of principal components used to transform the normalized expression matrix into PCA space.
        (default is 50)
    minkowski_order : int, optional
        The type of distance metric used:
        - 2 for Euclidean distance
        - 1 for Manhattan distance
        (default is 2)

    Outputs
    -------
    The function produces the mapping of cell type information between two modalities and saves the results in the specified output directory.

    """


    if output_nico_dir==None:
        outputdir='./nico_out/'
    else:
        outputdir=output_nico_dir
    create_directory(outputdir)

    ref_h5ad=refpath+'sct_singleCell.h5ad'
    que_h5ad=quepath+'sct_spatial.h5ad'
    #delimiter=','
    sct_ad_sp=sc.read_h5ad(que_h5ad)
    sct_ad_sc=sc.read_h5ad(ref_h5ad)

    original_h5ad=sc.read_h5ad(refpath+'Original_counts.h5ad')

    #cellname=np.reshape(cellname,(len(cellname),1))
    #annotation_singlecell_celltypename=np.reshape(annotation_singlecell_celltypename,(len(annotation_singlecell_celltypename),1))

    if output_annotation_dir==None:
        output_annot=outputdir+'annotations/'
    else:
        output_annot=output_annotation_dir
    create_directory(output_annot)
    #method='gauss'
    method='umap'
    adata_query=sc.read_h5ad(que_h5ad)

    sp_genename=sct_ad_sp.var_names.to_numpy()
    sc_genename=sct_ad_sc.var_names.to_numpy()
    index_sp,index_sc=find_index(sp_genename,sc_genename)

    ad_sp_ori=sct_ad_sp[:,index_sp].copy()
    ad_sc_ori=sct_ad_sc[:,index_sc].copy()

    ad_sc_ori.write_h5ad(output_annot+'final_sct_sc.h5ad')
    ad_sp_ori.write_h5ad(output_annot+'final_sct_sp.h5ad')

    fmnn=output_annot+"anchors_data_"+str(neigh)+'.npz'

    flag=1
    if os.path.isfile(fmnn):
        filesize = os.path.getsize(fmnn)
        # uncomment following part for not rewriting
        #if filesize>0:
        #    flag=0
    if flag==1:
        input_sp,input_sc,sp_barcode,sc_barcode=sct_return_sc_sp_in_shared_common_PC_space(ad_sp_ori,ad_sc_ori,no_of_pc,method)
        #print('sp',input_sp.shape,'\nsc',input_sc.shape)
        corrected = find_mutual_nn(minkowski_order,input_sp,input_sc,sp_barcode,sc_barcode, k1= neigh,k2= neigh)
        #pd.DataFrame(corrected).to_csv(fmnn,index=False,header=None)
        n_jobs=-1
        k_dist,k_index = cKDTree(input_sp).query(x=input_sp, k=neigh, p=minkowski_order,workers=n_jobs)
        knn_neigh=[]
        for i in range(len(k_index)):
            t=[]
            for j in range(len(k_index[i])):
                id=k_index[i,j]
                t.append(sp_barcode[id])
            knn_neigh.append(t)
        knn_neigh=np.array(knn_neigh)
        np.savez(fmnn,anchors=corrected,k_dist=k_dist,k_index=knn_neigh)

    inputvar={}
    inputvar['output_annot']=output_annot
    inputvar['KNN']=neigh
    inputvar['ad_sp']=ad_sp_ori
    inputvar['original_h5ad']=original_h5ad
    inputvar['adata_query']=adata_query
    inputvar['fmnn']=fmnn
    inputvar['output_nico_dir']=outputdir
    outputvar=SimpleNamespace(**inputvar)

    return outputvar

def delete_files(input):
    "This function will delete the anchors file and temporary file generated during the annotations."
    os.remove(input.output_annot+'final_sct_sc.h5ad')
    os.remove(input.output_annot+'final_sct_sp.h5ad')
    os.remove(input.fmnn)


def nico_based_annotation(previous,ref_cluster_tag='cluster',across_spatial_clusters_dispersion_cutoff=0.15,guiding_spatial_cluster_resolution_tag='leiden0.5',
number_of_iteration_to_perform_celltype_annotations=3,resolved_tie_issue_with_weighted_nearest_neighbor='No'):
    """
    **This function performs NiCo-based annotation of spatial cell transcriptomes by using label transfer from scRNAseq data.**

    The function utilizes label transfer to annotate spatial transcriptomic data based on cell type information from scRNAseq data. It leverages anchored cells to iteratively annotate non-anchor cells. The annotations are either performed using a majority vote or a weighted vote based on distances in the transformed gene expression space.

    The function starts by reading the output from `find_anchor_cells_between_ref_and_query` and annotates the spatial cells based on the provided parameters.

    Parameters
    ----------
    previous : object
        The output object from `find_anchor_cells_between_ref_and_query` containing necessary data for annotation.

    ref_cluster_tag : str, optional
        The slot in the reference anndata object file ('Original_counts.h5ad') where cell type information is stored <anndata>.obs[cluster].
        (default is 'cluster')

    across_spatial_clusters_dispersion_cutoff : float, optional
        The cutoff used to remove noisy anchors. Anchored cells that belong to any guiding spatial cluster with a frequency lower than this cutoff will be discarded.
        (default is 0.15)

    guiding_spatial_cluster_resolution_tag : str, optional
        The guiding spatial Leiden cluster resolution (clustering of spatial data used for anchor pruning). The `sct_spatial.h5ad` file should have required resolution of Leiden clusterings such as  0.3, 0.4, 0.5, 0.6, 0.7 and 0.8 that can be stored in the anndata.obs slot with name 'leiden0.3', 'leiden0.4', 'leiden0.5', 'leiden0.6', 'leiden0.7' and 'leiden0.8'.
        (default is 'leiden0.5')

    number_of_iteration_to_perform_celltype_annotations : int, optional
        The number of iterations to perform the cell type annotations. Higher numbers of iterations may annotate more cells but decrease confidence due to dilution of anchor information.
        (default is 3)

    resolved_tie_issue_with_weighted_nearest_neighbor : str, optional
        Whether to resolve tie issues in cell type assignment with a weighted nearest neighbor approach:
        - 'No': Assigns 'NM' (not mapped) to non-anchor cells in case of a tie.
        - 'Yes': Utilizes the weighted average of cell type proportions for resolving ties, with weights inversely proportional to the distance.
        (default is 'No')

    Outputs
    -------
    For each iteration, the function generates the following files in the specified output directory:
    - `_nico_annotation_cluster.csv`: Contains the annotated cluster information.
    - `_nico_annotation_ct_name.csv`: Contains the cell type names associated with each cluster.

    The default output directory for these files is `./nico_out/annotations/`.

    Notes
    -----
    - The niche function uses the final iteration of the annotations for finding niche cell type interactions in the `spatial_neighborhood_analysis`.
    """


    df=previous.original_h5ad.obs[ref_cluster_tag]
    annotation_singlecell_celltypename=df.to_numpy()
    cellname=df.index.to_numpy()

    ad_sc_ori=sc.read_h5ad(previous.output_annot+'final_sct_sc.h5ad')
    ad_sp_ori=sc.read_h5ad(previous.output_annot+'final_sct_sp.h5ad')
    singlecell_sct_barcode_id=ad_sc_ori.obs_names.to_numpy()
    spatialcell_sct_barcode_id=ad_sp_ori.obs_names.to_numpy()

    sc_ct_name=[]
    A=list(sorted(np.unique(annotation_singlecell_celltypename)))
    d={}
    for i in range(len(A)):
        sc_ct_name.append([i,A[i]])
        d[A[i]]=i
    sc_ct_name=np.array(sc_ct_name)
    #sc_cluster=np.hstack((cellname,annotation_singlecell_celltypename))
    sc_cluster=[]
    for j in range(len(annotation_singlecell_celltypename)):
        sc_cluster.append([cellname[j],d[annotation_singlecell_celltypename[j]]])
    sc_cluster=np.array(sc_cluster)
    annotation_singlecell_barcode_id=sc_cluster[:,0]
    annotation_singlecell_cluster_id=sc_cluster[:,1]
    singlecell_unique_clustername=sc_ct_name[:,1]
    singlecell_unique_clusterid=sc_ct_name[:,0]


    df=previous.adata_query.obs[guiding_spatial_cluster_resolution_tag]#.to_csv(spatialclusterFilename,header=True)
    annotation_spatial_barcode_id= df.index.to_numpy()
    annotation_spatial_cluster_id= df.to_numpy()
    spatialcell_unique_clustername=[]
    spatialcell_unique_clusterid=sorted(list(np.unique(annotation_spatial_cluster_id)))
    d={}
    for i in range(len(spatialcell_unique_clusterid)):
        name='c'+str(spatialcell_unique_clusterid[i])
        d[spatialcell_unique_clusterid[i]]=name
        spatialcell_unique_clustername.append(name)
    annotation_spatial_celltypename=[]
    for i in range(len(annotation_spatial_cluster_id)):
        annotation_spatial_celltypename.append(d[annotation_spatial_cluster_id[i]])
    annotation_spatial_celltypename=np.array(annotation_spatial_celltypename)
    spatialcell_unique_clustername=np.array(spatialcell_unique_clustername)



    data=np.load(previous.fmnn,allow_pickle=True)

    spatial_annotation_output_fname='nico_annotation'
    spatial_deg_annotation_output_clustername=spatial_annotation_output_fname+'_cluster.csv'
    spatial_deg_annotation_output_celltypename=spatial_annotation_output_fname+'_ct_name.csv'

    inputvar={}
    inputvar['fname_mnn_anchors']=data['anchors']
    inputvar['annotation_singlecell_barcode_id']=annotation_singlecell_barcode_id
    inputvar['annotation_singlecell_celltypename']=annotation_singlecell_celltypename
    inputvar['annotation_singlecell_cluster_id']=annotation_singlecell_cluster_id
    inputvar['lsc']=[singlecell_unique_clustername,singlecell_unique_clusterid]
    inputvar['sct_singlecell_barcode_id']=singlecell_sct_barcode_id
    inputvar['sct_spatial_barcode_id']=spatialcell_sct_barcode_id
    inputvar['annotation_spatial_barcode_id']=annotation_spatial_barcode_id
    inputvar['annotation_spatial_celltypename']=annotation_spatial_celltypename
    inputvar['annotation_spatial_cluster_id']=annotation_spatial_cluster_id
    inputvar['lsp']=[spatialcell_unique_clustername, spatialcell_unique_clusterid]
    inputvar['output_annot']=previous.output_annot
    inputvar['output_nico_dir']=previous.output_nico_dir
    inputvar['KNN']=previous.KNN
    inputvar['k_dist']=data['k_dist']
    inputvar['k_index']=data['k_index']
    inputvar['ad_sp']=previous.ad_sp
    inputvar['fmnn']=previous.fmnn
    inputvar['spatial_deg_annotation_output_clustername']=spatial_deg_annotation_output_clustername
    inputvar['spatial_deg_annotation_output_celltypename']=spatial_deg_annotation_output_celltypename
    inputvar['across_spatial_clusters_dispersion_cutoff']=across_spatial_clusters_dispersion_cutoff
    inputvar['number_of_iteration_to_perform_celltype_annotations']=number_of_iteration_to_perform_celltype_annotations
    inputvar['resolved_tie_issue_with_weighted_nearest_neighbor']=resolved_tie_issue_with_weighted_nearest_neighbor

    input=SimpleNamespace(**inputvar)
    good_anchors,visualize_anchors=find_commnon_MNN(input)
    inputvar['visualize_anchors']=visualize_anchors
    input=SimpleNamespace(**inputvar)



    chosenKNN=input.KNN
    sp_leiden_barcode2cluid={}
    sp_leiden_cluid2barcode={}
    for i in range(len(input.annotation_spatial_barcode_id)):
            id=input.annotation_spatial_cluster_id[i]
            name=input.annotation_spatial_barcode_id[i]
            sp_leiden_barcode2cluid[name]=id
    resolutionClusterWise=sp_leiden_barcode2cluid

    deg,G,weights=read_dist_and_nodes_as_graph(input.k_dist,input.k_index)
    mnn=input.fname_mnn_anchors

    index=[]
    for i in range(len(mnn)):
        name=mnn[i,0]+'#'+mnn[i,1]
        if name in good_anchors:
            index.append(i)
    mnn=mnn[index,:]

    sc_ctype_id=input.lsc[1]
    sc_ctype_name=input.lsc[0]


    a=np.reshape(input.annotation_singlecell_barcode_id,(len(input.annotation_singlecell_barcode_id),1))
    b=np.reshape(input.annotation_singlecell_cluster_id,(len(input.annotation_singlecell_cluster_id),1))

    sc_clusters=np.hstack((a,b))
    sp_cell_identity=find_all_the_spatial_cells_mapped_to_single_cells(sc_ctype_id,sc_clusters,mnn,sc_ctype_name)

    unique_mapped={}
    confused={}
    all_mapped={}
    for key in sp_cell_identity:
        name=''
        a=sp_cell_identity[key]
        #print('1' , a)
        for j in range(len(a)):
            name+='_a#d_'+a[j][0]
        if len(a)==1:
            unique_mapped[key]=a[0][0]
        else:
            t1=[]
            t2=[]
            for j in range(len(a)):
                t1.append(a[j][1])
                t2.append(a[j][0])
            confused[key]=t2
            #print(key,t1,t2)
        all_mapped[key]=name[5:]
        #fw.write(key+'\t'+str(name)+'\n')

    #print('unique mapped 1',len(unique_mapped))
    #fw=open(input.savepath+'unique_mapped.dat','w')
    #for key in unique_mapped:
    #    fw.write(key+'\t'+'0\n')
    #fw.close()

    ad_sp= input.ad_sp
    cellname=ad_sp.obs_names.to_numpy()
    genename=ad_sp.var_names.to_numpy()


    #saveunique_mapped=unique_mapped
    #unique_mapped=saveunique_mapped
    if (input.resolved_tie_issue_with_weighted_nearest_neighbor)=='No':
        all_anchored_mapped=resolved_confused_and_unmapped_mapping_of_cells_with_majority_vote(confused,G,all_mapped,unique_mapped,resolutionClusterWise)
    else:
        all_anchored_mapped=resolved_confused_and_unmapped_mapping_of_cells_with_weighted_average_of_inverse_distance_in_neighbors(confused,G,weights,all_mapped,unique_mapped,resolutionClusterWise)
    #print('unique mapped 2',len(all_anchored_mapped))
    availabled_anchors_mapped=all_anchored_mapped

    for iter in range(input.number_of_iteration_to_perform_celltype_annotations):
            unmapped_cellname,unmapped_deg=find_unmapped_cells_and_deg(deg,availabled_anchors_mapped)

            if (input.resolved_tie_issue_with_weighted_nearest_neighbor)=='No':
                unique_mapped=resolved_confused_and_unmapped_mapping_of_cells_with_majority_vote(unmapped_cellname,G,availabled_anchors_mapped,availabled_anchors_mapped,resolutionClusterWise)
            else:
                unique_mapped=resolved_confused_and_unmapped_mapping_of_cells_with_weighted_average_of_inverse_distance_in_neighbors(unmapped_cellname,G,weights,availabled_anchors_mapped,availabled_anchors_mapped,resolutionClusterWise)

            #print('iter',iter,len(unique_mapped),len(unmapped_cellname),len(unmapped_deg))

            for i in range(len(cellname)):
                key=cellname[i]
                if key not in unique_mapped:
                    unique_mapped[key]='NM'

            count=0
            availabled_anchors_mapped={}
            for key in unique_mapped:
                if unique_mapped[key]=='NM':
                    count+=1
                else:
                    availabled_anchors_mapped[key]=unique_mapped[key]
            #print('Iter',iter,count)

            deg_annot_cluster_fname=input.output_annot+str(iter+1)+'_'+input.spatial_deg_annotation_output_clustername
            deg_annot_ct_fname=input.output_annot+str(iter+1)+'_'+input.spatial_deg_annotation_output_celltypename

            nico_cluster=write_annotation(deg_annot_cluster_fname,deg_annot_ct_fname,unique_mapped,cellname)

    inputvar['nico_cluster']=nico_cluster
    inputvar['ad_sp_ori']=ad_sp_ori
    input=SimpleNamespace(**inputvar)

    return input


def read_dist_and_nodes_as_graph(knn_dist,knn_nodes):

    """
    Reads edges information from k-nearest neighbors (KNN) data and converts it into a graph representation.

    This helper function is used in the `nico_based_annotation` function to interpret the relationships between nodes (cells) based on their KNN distances. It constructs a graph `G` where nodes represent cells and edges represent the KNN relationships with associated distances as weights. Additionally, it calculates the degree of each node.

    Parameters
    ----------
    knn_dist : array-like
        A 2D array where each row represents a node and contains the distances to its k-nearest neighbors.

    knn_nodes : array-like
        A 2D array where each row represents a node and contains the indices of its k-nearest neighbors.

    Returns
    -------
    deg : dict
        A dictionary where keys are node indices and values are the degrees (number of edges) of the corresponding nodes.

    G : networkx.Graph
        A graph object where nodes represent cells and edges represent KNN relationships between cells.

    weights : dict
        A dictionary where keys are edge identifiers (formatted as 'node1#node2') and values are the distances between the nodes.

    Notes
    -----
    The function assumes that the first element in each row of `knn_nodes` is the node itself, and subsequent elements are its nearest neighbors.

    """

    weights={}
    for i in range(len(knn_nodes)):
        l=knn_nodes[i]
        dist=knn_dist[i]
        #print(l,dist)
        for n in range(1,len(l)):
            temp=sorted([l[0],l[n]])
            name=str(temp[0])+'#'+str(temp[1])
            weights[name]=dist[n]


    all_edges=[]
    for key in weights:
        name=key.split('#')
        all_edges.append(name)


    G=nx.Graph()
    G.add_edges_from(all_edges)
    deg = [d for (v, d) in G.degree()]
    nodes = [v for (v, d) in G.degree()]

    deg={}
    for (n,d) in G.degree:
        deg[n]=d

    return deg,G,weights

'''
def read_KNN_file(KNNfilename):

    f=open(KNNfilename)
    neighbors=[]
    for line in f:
        l=line[0:-1].split(',')
        neighbors.append(l[0:-1])
    edges=[]
    all_edges=[]
    d={}
    for j in range(len(neighbors)):
    #for j in range(1):
        l=neighbors[j]
        #for m in range(len(l)):
        for n in range(1,len(l)):
            temp=sorted([l[0],l[n]])
            name=temp[0]+'#'+temp[1]
            d[name]=1

            #all_edges.append([l[0].replace('cell',''),l[n].replace('cell','')])
    for key in d:
        name=key.split('#')
        #print(key,name)
        all_edges.append(name)


    G=nx.Graph()
    G.add_edges_from(all_edges)
    deg = [d for (v, d) in G.degree()]
    nodes = [v for (v, d) in G.degree()]

    deg={}
    for (n,d) in G.degree:
        deg[n]=d

    return deg,G
'''

def return_singlecells(cluster_data,midzone):
    """
    Finds the scRNAseq cells belonging to a specific cell type.

    This helper function is used in `find_all_the_spatial_cells_mapped_to_single_cells` to identify the single-cell RNA sequencing (scRNAseq) cells that belong to a specified cell type (midzone).

    Parameters
    ----------
    cluster_data : numpy.ndarray
        A 2D array where the first column contains barcode IDs and the second column contains cluster IDs.

    midzone : int or str
        The cluster ID representing the specific cell type of interest.

    Returns
    -------
    numpy.ndarray
        An array of unique barcode IDs corresponding to the cells that belong to the specified cell type (midzone).
    """

    barcode_id= cluster_data[:,0]
    cluster_id= cluster_data[:,1]
    index=np.where(cluster_id==midzone)
    midzoneCells=barcode_id[index[0]]
    return np.unique(midzoneCells)



def findSpatialCells(midzoneCells,mnn):
    """
    Finds the anchored cells for each cell type.

    This helper function is used in `find_all_the_spatial_cells_mapped_to_single_cells` to identify the spatial cells that are anchored to each single-cell RNA sequencing (scRNAseq) cell type.

    Parameters
    ----------
    midzoneCells : numpy.ndarray
        An array of barcode IDs representing the single-cell RNA sequencing (scRNAseq) cells belonging to a specific cell type.

    mnn : numpy.ndarray
        A 2D array where each row represents a mutual nearest neighbor (MNN) pair. The first column contains spatial cell barcode IDs and the second column contains scRNAseq cell barcode IDs.

    Returns
    -------
    dict
        A dictionary where keys are spatial cell barcode IDs and values are the counts of how many times each spatial cell is anchored to the scRNAseq cells.
    """
    d={}
    for i in range(len(midzoneCells)):
        first=midzoneCells[i]
        index=np.where(mnn[:,1]==first)
        spcells=mnn[index[0],0]
        #print(spcells)
        for k in range(len(spcells)):
            if spcells[k] not in d:
                d[spcells[k]]=1
            else:
                d[spcells[k]]+=1
    return d


def find_all_the_spatial_cells_mapped_to_single_cells(sc_ctype_id,sc_clusters,mnn,sc_ctype_name):
    """
    Maps spatial cells to single-cell RNA sequencing (scRNAseq) cell types.

    This helper function is used in `nico_based_annotation` to find the mapping of cells between spatial and scRNAseq modalities. It identifies the spatial cells that correspond to specific scRNAseq cell types based on mutual nearest neighbor (MNN) pairs.

    Parameters
    ----------
    sc_ctype_id : numpy.ndarray
        An array of unique identifiers for each scRNAseq cell type.

    sc_clusters : numpy.ndarray
        A 2D array where each row contains a barcode ID and a cluster ID representing the clustering of scRNAseq cells.

    mnn : numpy.ndarray
        A 2D array where each row represents an MNN pair. The first column contains spatial cell barcode IDs and the second column contains scRNAseq cell barcode IDs.

    sc_ctype_name : list of str
        A list of names corresponding to each scRNAseq cell type ID.

    Returns
    -------
    dict
        A dictionary where keys are spatial cell barcode IDs and values are lists containing the scRNAseq cell type names and their counts, with each spatial cell being mapped to the most frequent scRNAseq cell type if there are ties.

    """

    spdata=[]
    # single cell cluster id sc_ctype_id
    # single cell cluster name sc_ctype_name
    for i in range(len(sc_ctype_id)):
        sc_ct_specific_cells=return_singlecells(sc_clusters,sc_ctype_id[i])
        # all the single cell barcode id of sc_ctype_name[i]
        sp_ct_specific_cells=findSpatialCells(sc_ct_specific_cells,mnn)
        #print('1',i,sc_ctype_id[i],len(sp_ct_specific_cells))
        spdata.append(sp_ct_specific_cells)
        #print(sc_ctype_name[i], '\tSC',len(sc_ct_specific_cells),'\tSP',len(sp_ct_specific_cells))

    sp_cell_identity={}
    for i in range(len(sc_ctype_id)):
        a=spdata[i] # this is dictionary
        for name in a:
            if name not in sp_cell_identity:
                sp_cell_identity[name]=[[sc_ctype_name[i],a[name]]]
            else:
                sp_cell_identity[name].append([sc_ctype_name[i],a[name]])

    for key in sp_cell_identity:
        a=sp_cell_identity[key]
        #print(a)
        if len(a)>1:
            #print(key, a)
            t1=[]
            t2=[]
            for j in range(len(a)):
                t1.append(a[j][1])
                t2.append(a[j][0])
            ind=np.argsort(-np.array(t1))
            if t1[ind[0]]>t1[ind[1]]:
                b=[[t2[ind[0]],t1[ind[0]]]]
                sp_cell_identity[key]=b

        '''
        a=list(spdata[i])
        for j in range(len(a)):
            name=a[j]
            if name not in sp_cell_identity:
                sp_cell_identity[name]=[sc_ctype_name[i]]
            else:
                sp_cell_identity[name].append(sc_ctype_name[i])
        '''

    return sp_cell_identity

def write_annotation(deg_annot_cluster_fname,deg_annot_ct_fname,unique_mapped,cellname):
    """
    Generates CSV files for each iteration's annotation clusters and cell type names.

    This helper function is used in `nico_based_annotation` to create two CSV files:
    one for the cluster annotation and one for the cell type names with their frequencies.

    Parameters
    ----------
    deg_annot_cluster_fname : str
        The filename for the CSV file that will contain the cluster annotations.

    deg_annot_ct_fname : str
        The filename for the CSV file that will contain the cell type names and their frequencies.

    unique_mapped : dict
        A dictionary where keys are cell barcodes and values are their corresponding cell type names.

    cellname : numpy.ndarray
        An array of cell barcode IDs.

    Returns
    -------
    numpy.ndarray
        An array of cell type names corresponding to each cell barcode ID in the `cellname` array.

    """

    sc_ctype_name=[]
    d2={}
    for key in unique_mapped:
        a=unique_mapped[key]
        if a not in d2:
            d2[a]=1
        else:
            d2[a]+=1
        if a not in sc_ctype_name:
            sc_ctype_name.append(a)

    #print(sc_ctype_name)
    #print(d.keys())
    sc_ctype_name=sorted(sc_ctype_name)
    fw=open(deg_annot_ct_fname,'w')
    fw.write('clusterID,clusterName,Frequency\n')
    d={}
    for i in range(len(sc_ctype_name)):
        fw.write(str(i)+','+sc_ctype_name[i]+','+str(d2[sc_ctype_name[i]])+'\n')
        d[sc_ctype_name[i]]=i
    fw.close()

    #keys=sorted(list(unique_mapped.keys()))
    nico_cluster=[]
    fw=open(deg_annot_cluster_fname,'w')
    fw.write('barcode,mnn_based_annot\n')
    for i in range(len(cellname)):
        barcodeid=cellname[i]
        ctname=unique_mapped[barcodeid]
        fw.write(barcodeid+','+str(d[ctname])+'\n')
        nico_cluster.append(ctname)
    fw.close()

    return np.array(nico_cluster)



def find_unmapped_cells_and_deg(deg,unique_mapped):
    """
    Identifies unmapped non-anchored cells and their degrees.

    This helper function is used in `nico_based_annotation` to find cells that have not been mapped and their corresponding degree values.
    It returns the cell names and their degree values sorted in descending order of degree.

    Parameters
    ----------
    deg : dict
        A dictionary where keys are cell node identifiers and values are their corresponding degrees (number of connections).

    unique_mapped : dict
        A dictionary where keys are mapped cell node identifiers and values are their corresponding cell type names.

    Returns
    -------
    tuple : A tuple containing two numpy arrays:
        - cellname : numpy.ndarray
            An array of unmapped cell node identifiers sorted by their degree values in descending order.
        - degvalue : numpy.ndarray
            An array of degree values corresponding to the unmapped cell node identifiers, sorted in descending order.

    """

    un_mapped_nodes=[]
    un_mapped_deg=[]
    for node in deg:
        if node not in unique_mapped:
            un_mapped_nodes.append(node)
            un_mapped_deg.append(deg[node])

    un_mapped_deg=np.array(un_mapped_deg)
    un_mapped_nodes=np.array(un_mapped_nodes)
    index=np.argsort(-un_mapped_deg)

    cellname=un_mapped_nodes[index]
    degvalue=un_mapped_deg[index]

    return cellname,degvalue

'''
def resolved_confused_and_unmapped_mapping_of_cells_distance(confused,G,all_mapped):
    for mainkey in confused:
            a=G[mainkey]
            t=[]
            t1=[]
            for key in a:
                if key in all_mapped:
                    t.append(a[key]['weight'])
                    t1.append(key)
            t=np.array(t)
            t1=np.array(t1)
            ind=np.argsort(t)
            #print('4',len(t),t[ind])

            if len(t)>0:
                key=t1[ind[0]]
                t=[all_mapped[key]]
                t1=[]
                t2=[]
                t3=[]
                t4=[]
                for i in range(len(t)):
                    t1.append(t[i][2])
                    t2.append(t[i][0])
                    t3.append(t[i][1])
                    t4.append(t[i][3])
                ind=np.argsort(np.array(t1))
                finalone=[t2[ind[0]], t3[ind[0]],    t1[ind[0]] , t4[ind[0]]      ]
            else:
                finalone=['NM', -1,  99999999, 'Null'  ]
            #print('6',finalone1)

            all_mapped[mainkey]=finalone

    return all_mapped
'''

def resolved_confused_and_unmapped_mapping_of_cells_with_weighted_average_of_inverse_distance_in_neighbors(confused,G,weights,all_mapped,unique_mapped,sp_leiden_barcode2cluid_resolution_wise):
    """
    Annotates confused and unmapped spatial cells using a weighted average score from their neighbors.

    This helper function is used in `nico_based_annotation` to resolve the mapping of spatial cells that are either confused or not anchored by utilizing the weighted average of the inverse distance to their neighbors.

    Parameters
    ----------
    confused : list
        List of spatial cell identifiers that are confused and need to be resolved.

    G : networkx.Graph
        Graph where nodes represent cells and edges represent connections between cells.

    weights : dict
        A dictionary where keys are edge identifiers (formatted as 'node1#node2') and values are the corresponding weights (inverse of distances).

    all_mapped : dict
        Dictionary where keys are cell identifiers and values are their mapped cell types.

    unique_mapped : dict
        Dictionary to be updated where keys are cell identifiers and values are their resolved mapped cell types.

    sp_leiden_barcode2cluid_resolution_wise : dict
        Dictionary mapping cell identifiers to their cluster IDs based on Leiden clustering resolution.

    Returns
    -------
    dict
        Updated `unique_mapped` dictionary with resolved cell type annotations for the confused and unmapped cells.

    """
    for mainkey in confused:
            a=G[mainkey]
            #print('\n\n\n\n',mainkey,len(a))
            current_clu_id=sp_leiden_barcode2cluid_resolution_wise[mainkey]
            x=[]
            t=[]
            weight_score=[]
            for key in a:
                if key in all_mapped:
                    A=sorted([mainkey,key])
                    weight_score.append(1/weights[A[0]+'#'+A[1]]) #inverse of distance
                    t.append(all_mapped[key])
                    if current_clu_id==sp_leiden_barcode2cluid_resolution_wise[key]:
                        #x.append(sp_leiden_barcode2cluid_resolution_wise[key])
                        x.append(all_mapped[key])

            weighted_avg_score={}
            for i in range(len(t)):
                if t[i] not in weighted_avg_score:
                    weighted_avg_score[t[i]]=[weight_score[i]]
                else:
                    weighted_avg_score[t[i]].append(weight_score[i]) # are added


            if True:
                neigh_clu_id=list(np.unique(x)) #     Counter(x)
                c=Counter(t)
                totalsum=sum(c.values())
                #print('a',mainkey,len(t),c,totalsum)#confused[mainkey])
                #print('b',current_clu_id,neigh_clu_id)
                low2high=sorted(c, key=c.get)
                high2low=low2high[::-1]
                ws=[]
                tws=np.sum(weight_score)
                for key in high2low:
                    ws.append(np.sum(weighted_avg_score[key])/tws) #weighted average


                t1=[] #integer number to denote the degree for the negihboring cells who has similar cell type
                t2=[] #corresponding cell type name for from t1
                t3=[] #weighted score (inverse of distance)
                for i in range(len(high2low)):
                    if high2low[i].find('_a#d_')==-1:
                        t1.append(c[high2low[i]])
                        t2.append(high2low[i])
                        t3.append(ws[i])
                t1=np.array(t1)
                index=np.argsort(-t1)

                finalone='NM'
                for i in range(len(t1)):
                    localdeg=t1[index[i]]
                    localctname=t2[index[i]]
                    temp=[]
                    #This step checks cell type (highest degree to lowest) from neighbors to see whether they belong to the same guiding cluster or not
                    #only if the cell type of the majority of neighbors and current cell (confused or unresolved) are in the same guiding cluster will it be assigned to this cell type
                    if localctname in neigh_clu_id:
                        temp.append(localctname)
                    if len(np.unique(temp))==1:
                        finalone=temp[0]
                        break


                if len(t1)==0:
                    finalone='NM'# If no neighbor is found then it assigned to 'NM'

                if (finalone=='NM')&(len(t1)>0):
                    finalone='xxxx' #If you see this in the final annotation it means something is wrong
                    if len(t1)==1:
                        finalone=t2[0] #If neighbors belong to only one cell type then it assigned to that
                    elif t1[index[0]]>=t1[index[1]]: #If neighbors belong to many cell type (the first one has largest degree)
                        if t3[index[0]]<t3[index[1]]: #Here it preference for the normalized inverse distance score
                            finalone=t2[index[1]] #if the second cell type has lower degree than first but it has higher weight score then it choses the second cell type
                        else:
                            finalone=t2[index[0]] #if the second cell type has lower degree than first but it has lower weight score then it choses the first cell type


                    #print('b',current_clu_id,neigh_clu_id)
                    #print('index',index,t1[index])
                    #print('ok',high2low,c,ws)
                    #print('xx',neigh_clu_id,temp,finalone1)

                #print('final',finalone)
                unique_mapped[mainkey]=finalone
    return unique_mapped




def resolved_confused_and_unmapped_mapping_of_cells_with_majority_vote(confused,G,all_mapped,unique_mapped,sp_leiden_barcode2cluid):

    """
    Annotates confused anchored and non-anchored spatial cells using a majority vote scheme across the neighbors.

    This helper function is used in `nico_based_annotation` to resolve the mapping of spatial cells that are either confused or not anchored by utilizing the majority vote of their neighbors' annotations.

    Parameters
    ----------
    confused : list
       List of spatial cell identifiers that are confused and need to be resolved.

    G : networkx.Graph
       Graph where nodes represent cells and edges represent connections between cells.

    all_mapped : dict
       Dictionary where keys are cell identifiers and values are their mapped cell types.

    unique_mapped : dict
       Dictionary to be updated where keys are cell identifiers and values are their resolved mapped cell types.

    sp_leiden_barcode2cluid : dict
       Dictionary mapping cell identifiers to their cluster IDs based on Leiden clustering.

    Returns
    -------
    dict
       Updated `unique_mapped` dictionary with resolved cell type annotations for the confused and unmapped cells.

    """

    for mainkey in confused:
            a=G[mainkey]
            current_clu_id=sp_leiden_barcode2cluid[mainkey]
            x=[]
            t=[]
            for key in a:
                if key in all_mapped:
                    t.append(all_mapped[key])
                    if current_clu_id==sp_leiden_barcode2cluid[key]:
                            x.append(all_mapped[key])


            if True:
                neigh_clu_id=list(np.unique(x)) #     Counter(x)
                c=Counter(t)
                totalsum=sum(c.values())
                low2high=sorted(c, key=c.get)
                high2low=low2high[::-1]
                t2=[]
                t1=[]
                for i in range(len(high2low)):
                    if high2low[i].find('_a#d_')==-1:
                        t1.append(c[high2low[i]]) #integer number to denote the degree for the negihboring cells who has similar cell type
                        t2.append(high2low[i]) #corresponding cell type name for from t1
                t1=np.array(t1)
                index=np.argsort(-t1)

                finalone='NM'
                for i in range(len(t1)):
                    localdeg=t1[index[i]]
                    localctname=t2[index[i]]
                    #This step checks cell type (highest degree to lowest) from neighbors to see whether they belong to the same guiding cluster or not
                    #only if the cell type of the majority of neighbors and current cell (confused or unresolved) are in the same guiding cluster will it be assigned to this cell type
                    temp=[]
                    if localctname in neigh_clu_id:
                        temp.append(localctname)
                    if len(np.unique(temp))==1:
                        finalone=temp[0]
                        break

                if len(t1)==0:
                    finalone='NM'
                else:
                    finalone=finalone#t2[index]

                unique_mapped[mainkey]=finalone

    return unique_mapped


def visualize_umap_and_cell_coordinates_with_all_celltypes(output_annotation_dir=None,output_nico_dir=None,
anndata_object_name='nico_celltype_annotation.h5ad',
spatial_cluster_tag='nico_ct',spatial_coordinate_tag='spatial',umap_tag='X_umap',
number_of_iteration_to_perform_celltype_annotations=3,cmap=plt.cm.get_cmap('jet'),saveas='pdf',transparent_mode=False,showit=True,figsize=(15,6)):

    """
    Visualize UMAP and spatial coordinates with all cell types annotated in a single plot.

    This function generates visualizations for UMAP projections and spatial coordinates of cells, annotated by cell types. It saves the figures to specified directories and supports customization of various visualization parameters.

    Parameters:
    -----------
    output_annotation_dir : str, optional
        Directory to save the annotation figures. Default is './nico_out/annotations/'.

    output_nico_dir : str, optional
        Base directory for nico output files. Default is './nico_out/'.

    anndata_object_name : str, optional
        Name of the AnnData object file containing cell type annotations. Default is 'nico_celltype_annotation.h5ad'.

    spatial_cluster_tag : str, optional
        Key in AnnData object for spatial cluster annotations slot. Default is 'nico_ct'.

    spatial_coordinate_tag : str, optional
        Key in AnnData object for spatial coordinates slot. Default is 'spatial'.

    umap_tag : str, optional
        Key in AnnData object for UMAP embeddings slot. Default is 'X_umap'.

    number_of_iteration_to_perform_celltype_annotations : int, optional
        Number of iterations performed for cell type annotations. Default is 3.

    cmap : matplotlib.colors.Colormap, optional
        Colormap used to color the cell types. Default is 'jet'.

    saveas : str, optional
        Format to save the figures ('pdf' or 'png'). Default is 'pdf'.

    transparent_mode : bool, optional
        If True, sets the background color of the figures to transparent. Default is False.

    showit : bool, optional
        If True, displays the figures. Default is True.

    figsize : tuple, optional
        Dimensions of the figure size. Default is (15, 6).

    Outputs:
    --------
    Saves annotation figures to the following path './nico_out/annotations/'

    """



    if output_nico_dir==None:
        outputdir='./nico_out/'
    else:
        outputdir=output_nico_dir

    if output_annotation_dir==None:
        fig_save_path=outputdir+'annotations/'
    else:
        fig_save_path=output_annotation_dir


    #df_cluster=pd.read_csv(deg_annot_cluster_fname)
    #degbased_cluster=df_cluster.to_numpy()
    #df=pd.read_csv(deg_annot_ct_fname)
    #degbased_ctname=df.to_numpy()


    adata=sc.read_h5ad(outputdir+anndata_object_name)
    temp=adata.obsm[umap_tag]
    cellname=adata.obs_names.to_numpy()#df.index.to_numpy()
    cellname=np.reshape(cellname,(len(cellname),1))
    umap_data=np.hstack((cellname,temp))

    annot = adata.obs[spatial_cluster_tag]
    ctname = sorted(list(np.unique(annot)))
    degbased_ctname=[]
    d={}
    for i in range(len(ctname)):
        degbased_ctname.append([i,ctname[i]])
        d[ctname[i]]=i
    degbased_ctname=np.array(degbased_ctname)

    degbased_cluster=[]
    for i in range(len(cellname)):
        degbased_cluster.append([  adata.obs_names[i],d[annot[i]] ])
    degbased_cluster=np.array(degbased_cluster)


    #sometime if you have less number of spatial cells (due to filtering step) in the analysis than the position coordinate have
    #then need to find correct pairing.
    #umap_data=sort_index_in_right_order(degbased_cluster,umap_not_order)\

    #df=pd.read_csv(positionFilename)
    #posdata_not_order=df.to_numpy()
    posdata=np.hstack((cellname,adata.obsm[spatial_coordinate_tag]))
    #posdata=sort_index_in_right_order(degbased_cluster,posdata_not_order)

    points=np.zeros((len(posdata),2),dtype=float)
    location_cellname2int={}
    location_int2cellname={}
    for i in range(len(posdata)):
        name=posdata[i][0]
        location_cellname2int[name]=i
        location_int2cellname[i]=name
        points[i]=[posdata[i][1], posdata[i][2]]

    cellsinCT={}
    for i in range(len(degbased_cluster)):
        id=int(degbased_cluster[i][1])
        #celltype[degbased_cluster[i][0]]=id
        if id not in cellsinCT:
            cellsinCT[id]=[ location_cellname2int[ degbased_cluster[i][0]]]
        else:
            cellsinCT[id].append(location_cellname2int[ degbased_cluster[i][0]])


    CTname=degbased_ctname[:,1]

    fig,(ax)=plt.subplots(1,2,figsize=figsize)
    plot_all_ct(CTname,points,cellsinCT,ax[0],False,cmap)
    plot_all_ct(CTname,umap_data[:,[1,2]],cellsinCT,ax[1],True,cmap)
    ax[1].legend(loc='upper right',bbox_to_anchor=(1.50, 1),ncol=1, frameon=False,borderaxespad=0.,prop={"size":10},fancybox=True, shadow=True)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    #plt.gca().axes.get_yaxis().set_visible(False)
    ax[1].set_axis_off()

    #fig.tight_layout()
    print("The figures are saved: ", fig_save_path+'tissue_and_umap_with_all_celltype_annotations.'+saveas )
    fig.savefig(fig_save_path+'tissue_and_umap_with_all_celltype_annotations.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
    if showit:
        pass
    else:
        plt.close('all')


def save_annotations_in_spatial_object(inputdict,anndata_object_name='nico_celltype_annotation.h5ad'):
    """
    Save NiCo cell type cluster annotations in the AnnData object.

    This function takes a dictionary containing the necessary data and saves the cell type cluster annotations
    into the `.obs['nico_ct']` slot of an AnnData object. The updated AnnData object is then saved to a specified file.

    Inputs:

    inputdict : dict
        A dictionary containing the cell type annotations related objects.

    anndata_object_name : str, optional
        Name of the AnnData file to save the annotated data.
        Default is 'nico_celltype_annotation.h5ad'.

    Outputs:

    The function saves the annotated AnnData object in the specified directory ('./nico_out/') with the given file name.

    """
    print("Nico cell type cluster are saved in following path './nico_out/' as <anndata>.obs['nico_ct'] slot" )
    adata=inputdict.ad_sp_ori
    adata.obs['nico_ct']=inputdict.nico_cluster
    adata.write_h5ad(  inputdict.output_nico_dir+anndata_object_name)


def visualize_umap_and_cell_coordinates_with_selected_celltypes(
output_annotation_dir=None,output_nico_dir=None,
anndata_object_name='nico_celltype_annotation.h5ad',
spatial_cluster_tag='nico_ct',spatial_coordinate_tag='spatial',umap_tag='X_umap',
number_of_iteration_to_perform_celltype_annotations=3,choose_celltypes=[],msna=0.1,ms=0.5, showit=True,
cmap=plt.cm.get_cmap('jet'),saveas='pdf',transparent_mode=False,figsize=(8,3.5)):

    """
    Visualize UMAP and cell coordinates with selected cell types.

    This function visualizes the UMAP embedding and cell coordinates for selected cell types
    from spatial transcriptomics data.

    Inputs:

    output_annotation_dir : str, optional
        Directory path to save the annotation figures. Default is None, which uses './nico_out/annotations/'.

    output_nico_dir : str, optional
        Directory path for NiCo outputs. Default is None, which uses './nico_out/'.

    anndata_object_name : str, optional
        Name of the AnnData file containing cell type annotations. Default is 'nico_celltype_annotation.h5ad'.

    spatial_cluster_tag : str, optional
        Slot for spatial cluster annotations in the AnnData object. Default is 'nico_ct'.

    spatial_coordinate_tag : str, optional
        Slot for spatial coordinates in the AnnData object. Default is 'spatial'.

    umap_tag : str, optional
        Slot for UMAP embeddings in the AnnData object. Default is 'X_umap'.

    number_of_iteration_to_perform_celltype_annotations : int, optional
        Number of iterations for performing cell type annotations. Default is 3.

    choose_celltypes : list, optional
        List of cell types to visualize. Default is an empty list, which shows annotations for all cell types.

    msna : float, optional
        Marker size for non-selected (NA) cell types. Default is 0.1.

    ms : float, optional
        Marker size for selected cell types. Default is 0.5.

    showit : bool, optional
        Whether to display the figures. Default is True.

    cmap : Colormap, optional
        Colormap used to color the cell types. Default is plt.cm.get_cmap('jet').

    saveas : str, optional
        Format to save the figures ('pdf' or 'png'). Default is 'pdf'.

    transparent_mode : bool, optional
        Whether to use a transparent background for the figures. Default is False.

    figsize : tuple, optional
        Dimension of the figure size. Default is (8, 3.5).

    Outputs:

    The function saves individual cell type annotation figures in the specified directory.

    Notes:

    - Ensure that the input AnnData object contains the required tags for UMAP, spatial coordinates, and cell type annotations.
    - The function will save the annotation figures in the specified directory.
    """

    #df_cluster=pd.read_csv(deg_annot_cluster_fname)
    #degbased_cluster=df_cluster.to_numpy()
    #df=pd.read_csv(deg_annot_ct_fname)
    #degbased_ctname=df.to_numpy()

    if output_nico_dir==None:
        outputdir='./nico_out/'
    else:
        outputdir=output_nico_dir

    if output_annotation_dir==None:
        fig_save_path_main=outputdir+'annotations/'
    else:
        fig_save_path_main=output_annotation_dir

    adata=sc.read_h5ad(outputdir+anndata_object_name)
    temp=adata.obsm[umap_tag]
    cellname=adata.obs_names.to_numpy()#df.index.to_numpy()
    cellname=np.reshape(cellname,(len(cellname),1))
    umap_data=np.hstack((cellname,temp))
    posdata=np.hstack((cellname,adata.obsm[spatial_coordinate_tag]))

    annot = adata.obs[spatial_cluster_tag]
    ctname = sorted(list(np.unique(annot)))
    degbased_ctname=[]
    d={}
    for i in range(len(ctname)):
        degbased_ctname.append([i,ctname[i]])
        d[ctname[i]]=i
    degbased_ctname=np.array(degbased_ctname)

    degbased_cluster=[]
    for i in range(len(cellname)):
        degbased_cluster.append([  adata.obs_names[i],d[annot[i]] ])
    degbased_cluster=np.array(degbased_cluster)

    if len(choose_celltypes)==0:
        mycluster_interest_all=[]
        for fi in range(len(degbased_ctname)):
            CC_celltype_name=degbased_ctname[fi,1]
            mycluster_interest_all.append([CC_celltype_name])
    else:
        mycluster_interest_all=choose_celltypes


    mycluster_interest_id=[]
    for i in range(len(mycluster_interest_all)):
        temp=[]
        for k in range(len(mycluster_interest_all[i])):
            for j in range(len(degbased_ctname)):
                if degbased_ctname[j,1]==mycluster_interest_all[i][k]:
                    temp.append(degbased_ctname[j,0])
        mycluster_interest_id.append(temp)


    CTname=degbased_ctname[:,1]
    fig_save_path=fig_save_path_main+'fig_individual_annotation/'
    create_directory(fig_save_path)
    fig_save_path_leg=fig_save_path_main+'fig_individual_annotation/'+'leg/'
    create_directory(fig_save_path_leg)

    for fi in range(len(mycluster_interest_all)):
        CC_celltype_name=degbased_ctname[fi,1]+str(fi)
        mycluster_interest=mycluster_interest_all[fi]

        barcode=[]
        for j in range(len(mycluster_interest)):
            for i in range(len(degbased_ctname)):
                if degbased_ctname[i][1]==mycluster_interest[j]:
                    myindex=degbased_ctname[i][0]
            index=np.where(degbased_cluster[:,1]==myindex)
            barcode.append(degbased_cluster[index[0],0])


        points=np.zeros((len(posdata),2),dtype=float)
        location_cellname2int={}
        location_int2cellname={}
        for j in range(len(posdata)):
            name=posdata[j][0]
            location_cellname2int[name]=j
            location_int2cellname[j]=name
            points[j]=[posdata[j][1], posdata[j][2]]

        index=[]
        for j in range(len(barcode)):
            index.append([])

        for i in range(len(mycluster_interest_id[fi])):
            #print(i,'ankit',mycluster_interest_id[i])
            ind=np.where(degbased_cluster[:,1]==mycluster_interest_id[fi][i])
            index[i]=ind[0]

        cellsinCT={}
        for i in range(len(degbased_cluster)):
            id=int(degbased_cluster[i][1])
            #celltype[degbased_cluster[i][0]]=id
            if id not in cellsinCT:
                cellsinCT[id]=[ location_cellname2int[ degbased_cluster[i][0]]]
            else:
                cellsinCT[id].append(location_cellname2int[ degbased_cluster[i][0]])



        #=find_id(degbased_ctname,mycluster_interest,degbased_cluster)
        #PP2,id2,cellsinCT2=reading_data(posdata,cl2,degbased_cluster,mycluster_interest_id[i])
        cl2=barcode
        PP2=points
        id2=index



        fig,(ax)=plt.subplots(1,2,figsize=figsize)
        plot_specific_ct(mycluster_interest,PP2,id2,ax[0],cmap,ms,msna)
        plot_specific_ct(mycluster_interest,umap_data[:,[1,2]],id2,ax[1],cmap,ms,msna)

        leg1=ax[1].legend(loc='upper right',bbox_to_anchor=(1.50, 1),ncol=1, borderaxespad=0.,prop={"size":8},fancybox=True, frameon=False,shadow=True)

        ax[1].set_xticks([])
        ax[1].set_yticks([])
        #plt.gca().axes.get_yaxis().set_visible(False)
        ax[1].set_axis_off()
        fig.tight_layout()
        filename=fig_save_path+remove_extra_character_from_name(mycluster_interest[0])+str(fi)
        print("The figures are saved: ", filename+'.'+saveas)
        fig.savefig(filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
        if showit:
            pass
        else:
            plt.close('all')

        fig  = leg1.figure
        fig.canvas.draw()
        bbox  = leg1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        filename=fig_save_path_leg+remove_extra_character_from_name(mycluster_interest[0])+str(fi)
        fig.savefig(filename+'_leg'+'.'+saveas, bbox_inches=bbox,transparent=transparent_mode,dpi=300)
        if showit:
            pass
        else:
            plt.close('all')




def remove_extra_character_from_name(name):
    """
    Remove special characters from cell type names to avoid errors while saving figures.

    This function replaces certain special characters in the input `name` with
    underscores or other appropriate characters to ensure the name is safe for use
    as a filename.

    Parameters
    ----------
    name : str
        The original cell type name that may contain special characters.

    Returns
    -------
    str
        The modified cell type name with special characters removed or replaced.

    Example
    -------
    >>> name = 'T-cell (CD4+)/CD8+'
    >>> clean_name = remove_extra_character_from_name(name)
    >>> print(clean_name)
    'T-cell_CD4p_CD8p'

    Notes
    -----
    The following replacements are made:

        - '/' is replaced with '_'
        - ' ' (space) is replaced with '_'
        - '"' (double quote) is removed
        - "'" (single quote) is removed
        - ')' is removed
        - '(' is removed
        - '+' is replaced with 'p'
        - '-' is replaced with 'n'
        - '.' (dot) is removed

    These substitutions help in creating filenames that do not contain characters
    that might be problematic for file systems or software.
    """
    name=name.replace('/','_')
    name=name.replace(' ','_')
    name=name.replace('"','')
    name=name.replace("'",'')
    name=name.replace(')','')
    name=name.replace('(','')
    name=name.replace('+','p')
    name=name.replace('-','n')
    name=name.replace('.','')
    return name


def plot_all_ct(CTname,PP,cellsinCT,ax,flag,cmap):
    """
    Helper function used in visualize_umap_and_cell_coordinates_with_all_celltypes to plot all cell types together.

    This function plots the locations of all cell types together on a single plot, with each cell type assigned a different color. Optionally, it can label each cell type with its index.

    Parameters
    ----------
    CTname : list of str
        A list of cell type names to be plotted.
    PP : numpy.ndarray
        A 2D numpy array of shape (n_cells, 2) containing the coordinates of the cells.
    cellsinCT : list of list of int
        A list where each element is a list of indices corresponding to cells of a specific type.
    ax : matplotlib.axes.Axes
        The axes object where the plot will be drawn.
    flag : bool
        A flag indicating whether to label each cell type with its index on the plot.
    cmap : matplotlib.colors.Colormap
        The colormap used to assign colors to different cell types.

    """
    #cmap=plt.cm.get_cmap('Spectral')
    #cmap=plt.cm.get_cmap('jet')

    cumsum=np.linspace(0,1,len(CTname))

    for i in range(len(CTname)):
        index=cellsinCT[i]
        labelname=str(i)+'-'+CTname[i]+'-'+str(len(index))
        rgba=cmap(cumsum[i])
        ax.plot(PP[index,0],PP[index,1],'o',label=labelname,color=rgba,markersize=1)
        x=np.mean(PP[index,0])
        y=np.mean(PP[index,1])
        if flag:
            ax.text(x,y,str(i),fontsize=12)


def plot_specific_ct(CTname,PP,index,ax,cmap,ms,msna):
    """
    Plots individual cell types for visualizing cell type annotations.

    This helper function is used to visualize_umap_and_cell_coordinates_with_selected_celltypes by plotting the locations of cells belonging to specific cell types. Cells not belonging to any specified cell types are also plotted in a different color.

    Parameters
    ----------
    CTname : list of str
        A list of cell type names to be plotted.
    PP : numpy.ndarray
        A 2D numpy array of shape (n_cells, 2) containing the coordinates of the cells.
    index : list of list of int
        A list where each element is a list of indices corresponding to cells of a specific type.
    ax : matplotlib.axes.Axes
        The axes object where the plot will be drawn.
    cmap : matplotlib.colors.Colormap
        The colormap used to assign colors to different cell types.
    ms : int or float
        The marker size for plotting the cells of specified types.
    msna : int or float
        The marker size for plotting the cells not belonging to any specified types.

    Returns
    -------
    None
        This function does not return any value. It directly plots on the provided axes object.

    """
    cumsum=np.linspace(0,1,len(CTname))
    remaining_index=[]
    for i in range(len(PP)):
        for j in range(len(index)):
            if i not in index[j]:
                remaining_index.append(i)

    ax.plot(PP[remaining_index,0],PP[remaining_index,1],'.',color="0.5",label='NA',markersize=msna)
    for j in range(len(index)):
        labelname=CTname[j]+'-'+str(len(index[j]))  #str(j)+'-'+
        #labelname=str(j)+'-'+CTname[j]
        rgba=cmap(cumsum[j])
        ax.plot(PP[index[j],0],PP[index[j],1],'o',label=labelname,color=rgba,markersize=ms)
        x=np.mean(PP[index[j],0])
        y=np.mean(PP[index[j],1])
        #ax.text(x,y,str(j),fontsize=12)

'''
def sort_index_in_right_order(correct,wrong):
    """
    Sorts the 'wrong' array to match the order of the 'correct' array based on the first column.

    This helper function is used to visualize cell type annotations by ensuring that the indices in the 'wrong' array are ordered to match the 'correct' array. The matching is done based on the values in the first column of both arrays.

    Parameters
    ----------
    correct : numpy.ndarray
        A 2D numpy array where the order of elements in the first column is the desired order.
    wrong : numpy.ndarray
        A 2D numpy array where the elements in the first column need to be reordered to match the 'correct' array.

    Returns
    -------
    numpy.ndarray
        A reordered version of the 'wrong' array where the first column matches the order of the 'correct' array.

    Notes
    -----
    - The function assumes that both 'correct' and 'wrong' are 2D numpy arrays with the same set of unique values in their first columns.
    - The function creates a dictionary to map values from the first column of 'wrong' to their indices, and then uses this mapping to reorder 'wrong' to match 'correct'.
    """
    Examples
    "Helper function used to visualize cell type annotations."
    d={}
    for i in range(len(wrong)):
        d[wrong[i,0]]=i
    index=[]
    for i in range(len(correct)):
        index.append(d[correct[i,0]])
    right=wrong[index]
    return right
'''
