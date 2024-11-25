

from scipy.spatial import Voronoi, ConvexHull,voronoi_plot_2d, Delaunay

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_predict, cross_val_score,RepeatedStratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import make_scorer,accuracy_score, f1_score, classification_report,confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
#from sklearn.metrics import precision_recall_fscore_support as score
#from imblearn.over_sampling import SMOTE, SMOTEN,ADASYN, KMeansSMOTE, SVMSMOTE
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc

#Metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import matthews_corrcoef

import pandas as pd
import numpy as np
import seaborn as snn
import scanpy as sc
import os
import random
import warnings
#import time
#import pickle5 as pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import networkx as nx
import pickle
from types import SimpleNamespace
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
#export PYTHONWARNINGS='ignore:Multiprocessing-backed parallel loops:UserWarning'
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"





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

def findNeighbors_in_given_radius(location,radius):
    """
    Find the neighbors for each cell using the given radius.

    This helper function used in ``create_spatial_CT_feature_matrix`` identifies the neighboring cells for each cell within the specified
    radius and computes the average distance to these neighbors.

    Parameters:
    -----------
    location : np.ndarray
        An array of shape (n, 3) representing the coordinates of the cells.
    radius : float
        The radius within which to search for neighboring cells.
        For immediate neighbors it is 0

    Returns:
    --------
    list
        A list of lists where each sublist contains the indices of the neighbors for each cell.
    """

    n=location.shape[0]
    #print('ss',location.shape)
    neighbor={}
    mydist=0
    mycount=0
    for i in range(n):
        loc1=location[i]
        #print(i,loc1)
        t1=(loc1[0]-1.1*radius) <= location[:,0]
        t2=location[:,0] <= (loc1[0]+1.1*radius)
        t3=(loc1[1]-1.1*radius) <= location[:,1]
        t4=location[:,1] <= (loc1[1]+1.1*radius)
        t5=(loc1[2]-1.1*radius) <= location[:,2]
        t6=location[:,2] <= (loc1[2]+1.1*radius)

        index=  np.where ( t1 & t2 & t3 & t4 & t5 & t6    )
        count=0

        for k in range(len(index[0])):
            j=index[0][k]
            if j!=i:
                count+=1
                loc2=location[j]
                dist=euclidean_dist(loc1,loc2)
                flag=0
                if dist<radius:
                    if i not in neighbor:
                        neighbor[i]=[j]
                        flag=1
                    else:
                        if j not in neighbor[i]:
                            neighbor[i].append(j)
                            flag=1

                    if j not in neighbor:
                        neighbor[j]=[i]
                        flag=1
                    else:
                        if i not in neighbor[j]:
                            neighbor[j].append(i)
                            flag=1

                    if flag==1:
                        mydist=mydist+dist
                        mycount=mycount+1

        #print('t',count,len(index[0]))


    newneig=[]
    avg_neigh=0.0
    for i in range(n):
        try:
            l=neighbor[i]
        except KeyError:
            l=[]
        #print(l)
        newneig.append(l)
        avg_neigh+=len(l)

    print('average neighbors:',avg_neigh/n)
    print('average distance:',mydist/mycount)

    return newneig


def find_neighbors(pindex, triang):
    """
    Find the neighbors for a given point index using Delaunay triangulation.

    This helper function used in ```create_spatial_CT_feature_matrix``` identifies the neighboring points (cells) for a given point index
    using the Delaunay triangulation.

    Parameters:
    -----------
    pindex : int
        The index of the point (cell) for which neighbors are to be found.
    triang : scipy.spatial.Delaunay
        The Delaunay triangulation of the point set.

    Returns:
    --------
    np.ndarray
        An array of indices representing the neighbors of the given point.
    """

    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]


def create_spatial_CT_feature_matrix(radius,PP,louvain,noct,fraction_CT,saveSpatial,epsilonThreshold):
    """
    Generate the expected spatial cell type neighborhood matrix.

    This helper function is used in spatial_neighborhood_analysis to create a matrix that represents
    the expected neighborhood cell type composition based on spatial data. It uses either a radius-based
    approach or Delaunay triangulation to determine neighboring cells.

    Parameters
    ----------
    radius : float
        Radius within which to find neighbors. If set to 0, Delaunay triangulation is used instead.
    PP : np.ndarray
        Array of spatial coordinates of cells. Shape (n_cells, n_dimensions).
    louvain : np.ndarray
        Array containing Louvain clustering results for each cell. Shape (n_cells, 1).
    noct : int
        Number of cell types.
    fraction_CT : list of float
        List representing the fraction of each cell type.
    saveSpatial : str
        Path to save the output file containing the normalized spatial neighborhood matrix.
    epsilonThreshold : float
        Threshold distance cutoff to limit the distant neighbors when using Delaunay triangulation.

    Returns
    -------
    tuple
        - M : int
            Placeholder for future calculations (currently always returns 0).
        - neighbors : list of list of int
            List of neighbors for each cell. Each sublist contains indices of neighboring cells.
        - distance : list of list of float
            List of distances to neighbors for each cell. Each sublist contains distances to the neighboring cells.

    Notes
    -----
    - If radius is set to 0, Delaunay triangulation is used to find the neighbors within the epsilonThreshold distance.
    - The function saves the normalized spatial neighborhood matrix as a .npz file at the specified location.
    """


    if radius==0:
        value=np.sum(PP,axis=0)
        if (PP.shape[1]==3)&(value[2]==0):
            PP=PP[:,[0,1]]
        tri = Delaunay(PP)
        newneig=[]
        avg_neigh=0.0
        mydist=0
        mycount=0
        for i in range(len(PP)):
            ol = find_neighbors(i,tri)
            l=[]
            temp=[]
            for j in range(len(ol)):
                dist=euclidean_dist(PP[i],PP[ol[j]])
                temp.append(dist)
                if dist<epsilonThreshold:
                    l.append(ol[j])
                    mydist=mydist+dist
                    mycount=mycount+1
                #print(i,j,dist,PP[i],PP[l[j]])
            #print(PP[i,l)
            #if len(l)!=len(ol):
            #    print(i,type(ol),ol,l,temp,'\n')
            newneig.append(l)
            avg_neigh+=len(l)

        print('average neighbors:',avg_neigh/len(PP))
        print('average distance:',mydist/mycount)
        neighbors=newneig

    else:
        neighbors=findNeighbors_in_given_radius(PP,radius)
    n=len(neighbors)

    expectedNeighbors=[]
    input_mat_for_log_reg=[]
    for i in range(n):
        cell1=i
        CT1=louvain[i,0]
        V=neighbors[i]
        temp=[]
        CT2=np.zeros(len(noct),dtype=float)
        for j in range(len(V)):
            name=louvain[V[j],0]
            try:
                CT2[name]+=1.0
            except KeyError:
                pass
        #fw.write(str(cell1)+'\t'+str(CT1))
        temp.append(cell1)
        temp.append(CT1)
        expected=np.array(fraction_CT)*np.sum(CT2)
        tt=CT1
        #print(np.concatenate(np.array(celltype[key]),CT2))
        expectedNeighbors.append(np.concatenate([np.asarray([tt]),CT2]))
        #print(expectedNeighbors)
        CT2=CT2/expected   #np.sum(CT2) #np.linalg.norm(CT2)
        for j in CT2:
            #fw.write('\t'+'%0.5f'%j)
            temp.append(j)
        #fw.write('\n')
        input_mat_for_log_reg.append(temp)

    input_mat_for_log_reg=np.array(input_mat_for_log_reg)
    np.savez(saveSpatial+'normalized_spatial_neighborhood_'+str(radius)+'.npz',input_mat_for_log_reg=input_mat_for_log_reg)

    M=0

    distance=[]
    for i in range(len(neighbors)):
        cc=louvain[i,0]
        p1=PP[i]
        temp=[]
        for j in range(len(neighbors[i])):
            nid=neighbors[i][j]
            nc=louvain[nid,0]
            p2=PP[nid]
            dist=euclidean_dist(p1,p2)
            #print(p1,p2,p1,nid,nc,dist)
            temp.append(dist)
            #print(i,cc,j,nid,p1,p2,dist)
        distance.append(temp)


    return M, neighbors,distance





def euclidean_dist(p1,p2):
    "Calculate euclidean distance between two points in 2d/3d."
    if len(p1)==2:
        value=np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
    if len(p1)==3:
        value=np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2  + (p1[2]-p2[2])**2)
    return value


def reading_data(coordinates,louvainFull,degbased_ctname,saveSpatial,removed_CTs_before_finding_CT_CT_interactions):
    """
    Helper function used in spatial_neighborhood_analysis to read the cell coordinate file,
    cluster file, and cluster name file according to the input cell type list provided for the prediction.

    Parameters:
    -----------
    coordinates : str
        Path to the file containing cell coordinates.

    louvainFull : str
        Path to the file containing the full louvain clustering information.

    degbased_ctname : list of tuples
        A list where each element is a tuple containing the cell type ID and the cell type name.

    saveSpatial : str
        Path where the spatial analysis results should be saved.

    removed_CTs_before_finding_CT_CT_interactions : list of str
        A list of cell type names that should be excluded from the analysis.

    Returns:
    --------
    CTname : list of str
        A list of cell type names that are included in the analysis after filtering out the removed cell types.

    CTid : list of int
        A list of cell type IDs corresponding to the filtered cell type names.


    Notes:
    ------
    - This function assumes that `degbased_ctname` is a list of tuples where the first element is an integer
      representing the cell type ID and the second element is a string representing the cell type name.
    - The function filters out the cell types listed in `removed_CTs_before_finding_CT_CT_interactions` from the
      `degbased_ctname` list and returns the remaining cell type names and IDs.
    """
    #df=pd.read_csv(celltypeFilename)
    #data=df.to_numpy()

    CTname=[]
    CTid=[]
    for i in range(len(degbased_ctname)):
        name=degbased_ctname[i][1]
        if name not in removed_CTs_before_finding_CT_CT_interactions:
            CTname.append(name)
            CTid.append(degbased_ctname[i][0])


    #CTid=data[:,0]
    #CTname=data[:,1]

    #df=pd.read_csv(clusterFilename,sep=delimiter)
    #louvainFull=df.to_numpy()

    celltype={}
    cellsinCT={}
    index=[]
    for i in range(len(louvainFull)):
        clu_id=louvainFull[i][1]
        cel_id=louvainFull[i][0]
        if clu_id in CTid:
            index.append(i)
            #celltype[cel_id]=clu_id
            if clu_id not in cellsinCT:
                cellsinCT[clu_id]=[cel_id]
            else:
                cellsinCT[clu_id].append(cel_id)

    louvain=louvainFull[index,:]
    points=coordinates
    #need to sort the coordinates according to louvain order of cells
    temp={}
    for i in range(len(points)):
        temp[points[i,0]]=i

    index=[]
    for i in range(len(louvain)):
        id=louvain[i][0]
        index.append(temp[id])

    PP=points[index,:]

    location_cellname2int={}
    location_int2cellname={}
    for i in range(len(PP)):
        name=PP[i,0]
        location_cellname2int[name]=i
        location_int2cellname[i]=name

    #for 2d system
    if PP.shape[1]==3:
        points=np.zeros((PP.shape[0],3),dtype=float)
        points[:,0]=PP[:,1]
        points[:,1]=PP[:,2]
        PP=points
        tissue_dim=2

    #for 3d system
    if PP.shape[1]==4:
        PP=PP[:,1:]
        tissue_dim=3


    #print('louvain',louvain.shape,PP.shape)

    noct=sorted(cellsinCT)
    actual_count=[]
    fraction_CT=[]
    for key in noct:
        actual_count.append(len(cellsinCT[key]))
        fraction_CT.append(len(cellsinCT[key])/float(len(louvainFull)))

    #print('no of cell types',len(noct))

    temp=np.where(np.array(actual_count)>=5)
    good_index_cell_counts=temp[0]
    #print(actual_count,noct[good_index_cell_counts])

    less_no_cells_remove=[]
    for i in range(len(good_index_cell_counts)):
        index=np.where(louvain==noct[good_index_cell_counts[i]])
        less_no_cells_remove+=list(index[0])

    #print(less_no_cells[0:10],len(louvain))
    less_no_cells_remove=sorted(less_no_cells_remove)


    PP=PP[less_no_cells_remove]
    louvain=louvain[less_no_cells_remove]
    #print('louvain',louvain.shape,PP.shape)

    new_CT_id={}
    for i in range(len(good_index_cell_counts)):
        new_CT_id[noct[good_index_cell_counts[i]]]=i
    #print('a',np.unique(louvain))
    for i in range(len(louvain)):
        value=louvain[i,1]
        louvain[i,1]=new_CT_id[value]
        #print(value,louvain[i])

    fw=open(saveSpatial+'used_CT.txt','w')
    for i in range(len(new_CT_id)):
            value=fraction_CT[good_index_cell_counts[i]]
            name=CTname[good_index_cell_counts[i]]
            #print(CTname[key], key, total,len(cellsinCT[key]))
            fw.write(str(i)+'\t'+name+'\t'+str('%0.4f'%value)+'\n')
    fw.close()

    louvainWithBarcodeId=louvain
    louvain=louvain[:,1:]

    return PP, louvain, noct,fraction_CT,louvainWithBarcodeId




def plot_multiclass_roc(clf, X_test, y_test, n_classes):
    """
    Compute the ROC (Receiver Operating Characteristic) curve for each cell type prediction
    and evaluate its performance on the test dataset.

    Parameters:
    -----------
    clf : classifier object
        The classifier used for making predictions. It should have a `decision_function` method.

    X_test : array-like of shape (n_samples, n_features)
        Test feature set.

    y_test : array-like of shape (n_samples,)
        True labels for the test set.

    n_classes : int
        Number of unique classes (cell types) in the dataset.

    Returns:
    --------
    fpr : dict
        A dictionary where the keys are class indices and the values are arrays of false positive rates.

    tpr : dict
        A dictionary where the keys are class indices and the values are arrays of true positive rates.

    roc_auc : dict
        A dictionary where the keys are class indices and the values are the area under the ROC curve (AUC) scores.

    Notes:
    ------
    - This function uses the `decision_function` method of the classifier to get the confidence scores for each class.
    - The true labels `y_test` are converted into a binary format using one-hot encoding.
    - The ROC curve is computed for each class and the AUC score is calculated for each ROC curve.
    """
    y_score = clf.decision_function(X_test)
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def plot_confusion_matrix(input,saveas='pdf',showit=True,transparent_mode=False,dpi=300,figsize=(5.5,5)):
    """
    Generate and save a confusion matrix plot from the results of spatial_neighborhood_analysis.

    Parameters:
    -----------
    input : dict, or similar object
        The main input is the output from spatial_neighborhood_analysis.

    saveas : str, optional, default='pdf'
        Format to save the figure. Options are 'pdf' or 'png'. If 'png', the dpi is set to 300.

    showit : bool, optional, default=True
        Whether to display the plot after saving. If False, the plot will be closed after saving.

    transparent_mode : bool, optional, default=False
        Whether to save the figure with a transparent background.

    figsize : tuple of float, optional, default=(5.5, 5)
        Size of the figure in inches.

    Outputs:
    --------
    The function saves the confusion matrix plot in the directory specified by `nico_out/niche_prediction_linear/`.
    The filename will be in the format 'Confusing_matrix_R<Radius>.<saveas>', where <Radius> is the radius value
    from the input and <saveas> is the file format.

    Notes:
    ------
    - The function loads data from a numpy file specified by `input.fout`, which should contain the confusion matrix
      and related data.
    - The confusion matrix is plotted using seaborn's heatmap function with annotations.
    - The plot is saved in the specified format and directory, and optionally displayed based on the `showit` parameter.
    """


    data=np.load(input.fout,allow_pickle=True)
    coef=data['coef']
    cmn=data['cmn']
    cmn_std=data['cmn_std']
    coef_std=data['coef_std']
    CTFeatures=data['CTFeatures']

    filename='R'+str(input.Radius)

    plt.figure(figsize=figsize)
    classNames=[]
    for i in range(len(input.classes)):
        classNames.append(input.nameOfCellType[input.classes[i]])
    #print(classes,nameOfCellType,inputFeatures)
    snn.heatmap(cmn,annot=True, fmt='.2f',xticklabels=classNames, annot_kws={"size": 3},yticklabels=classNames)
    plt.xlabel('Predicted classes')
    plt.ylabel('Truth classes')
    plt.title('R = '+str(input.Radius)+', C='+str(input.lambda_c))
    plt.tight_layout()
    print("The figures are saved: ", input.niche_pred_outdir+'Confusing_matrix_'+filename+'.'+saveas)
    plt.savefig(input.niche_pred_outdir+'Confusing_matrix_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)
    if showit:
        pass
    else:
        plt.close('all')

def plot_coefficient_matrix(input,saveas='pdf',showit=True,transparent_mode=False,dpi=300,figsize=(5,8)):
    """
    Generate and save a coefficient matrix plot from the results of spatial_neighborhood_analysis.

    Parameters:
    -----------
    input : dict, or similar object
            The main input is the output from spatial_neighborhood_analysis.

    saveas : str, optional, default='pdf'
        Format to save the figure. Options are 'pdf' or 'png'. If 'png', the dpi is set to 300.

    showit : bool, optional, default=True
        Whether to display the plot after saving. If False, the plot will be closed after saving.

    transparent_mode : bool, optional, default=False
        Whether to save the figure with a transparent background.

    figsize : tuple of float, optional, default=(5, 8)
        Size of the figure in inches.

    Outputs:
    --------
    The function saves the coefficient matrix plot in the directory specified by ./nico_out/niche_prediction_linear/.
    The filename will be in the format "weight_matrix_R<Radius>.<saveas>", where <Radius> is the radius value
    from the input and <saveas> is the file format.

    """


    data=np.load(input.fout,allow_pickle=True)
    coef=data['coef']
    cmn=data['cmn']
    cmn_std=data['cmn_std']
    coef_std=data['coef_std']
    CTFeatures=data['CTFeatures']
    classNames=[]
    for i in range(len(input.classes)):
        classNames.append(input.nameOfCellType[input.classes[i]])

    filename='R'+str(input.Radius)
    plt.figure(figsize=figsize)
    #plt.figure()
    #snn.set(font_scale=0.4)
    b=snn.heatmap(coef.transpose(),yticklabels=CTFeatures,xticklabels=classNames)
    #plt.xticks(rotation=90)
    _, ylabels= plt.yticks()
    b.set_yticklabels(ylabels, size = 5)

    if input.BothLinearAndCrossTerms==1:
        plt.ylabel('Features linear  terms')
    else:
        plt.ylabel('Features cross terms')
    #plt.xlabel('# of classes (no of cell types)')
    plt.title('R = '+str(input.Radius)+', C='+str(input.lambda_c))
    plt.tight_layout()
    print("The figures are saved: ", input.niche_pred_outdir+'weight_matrix_'+filename+'.'+saveas)
    plt.savefig(input.niche_pred_outdir+'weight_matrix_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)
    if showit:
        pass
    else:
        plt.close('all')


def plot_predicted_probabilities(input,saveas='pdf',showit=True,transparent_mode=False,dpi=300,figsize=(12,6)):
    """
    Generate and save a plot of predicted probabilities from the results of spatial_neighborhood_analysis.

    Parameters:
    -----------
    input : dict, or similar object
            The main input is the output from spatial_neighborhood_analysis.

    saveas : str, optional, default='pdf'
        Format to save the figure. Options are 'pdf' or 'png'. If 'png', the dpi is set to 300.

    showit : bool, optional, default=True
        Whether to display the plot after saving. If False, the plot will be closed after saving.

    transparent_mode : bool, optional, default=False
        Whether to save the figure with a transparent background.

    figsize : tuple of float, optional, default=(12, 6)
        Size of the figure in inches.

    Outputs:
    --------
    The function saves the plot of predicted probabilities in the directory specified by ./nico_out/niche_prediction_linear/.
    The filename will be in the format 'predicted_probability_R<Radius>.<saveas>', where <Radius> is the radius value
    from the input and <saveas> is the file format.

    """


    filename='R'+str(input.Radius)

    plt.figure(figsize=figsize)

    plt.subplot(1,3,1)
    snn.heatmap(input.x_train,xticklabels=input.inputFeatures)
    plt.xlabel('# of input Features')
    plt.title('training set')
    plt.ylabel('75% of data')

    plt.subplot(1,3,2)
    snn.heatmap(input.x_test,xticklabels=input.inputFeatures)
    plt.xlabel('# of input Features')
    plt.title('testing set')
    plt.ylabel('25% of data')

    plt.subplot(1,3,3)
    snn.heatmap(input.predicted_probs,xticklabels=input.classes)
    plt.title('Predicted probability')
    plt.xlabel('# of classes (no of cell types)')
    plt.tight_layout()
    print("The figures are saved: ", input.niche_pred_outdir+'predicted_probability_'+filename+'.'+saveas)
    plt.savefig(input.niche_pred_outdir+'predicted_probability_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)

    if showit:
        pass
    else:
        plt.close('all')
    #print(predicted_probs)
    #prob=sigmoid( np.dot([y_train, y_test,1], log_reg_model.coef_.T) + log_reg_model.intercept_ )
    #print(prob)


def plot_roc_results(input,nrows=4,ncols=4,saveas='pdf',showit=True,transparent_mode=False,dpi=300,figsize=(10, 7),):
    """
    Generate and save ROC curves for the top 16 cell type predictions from the results of spatial_neighborhood_analysis.

    Parameters:
    -----------
    input : dict, or similar object
        The main input is the output from spatial_neighborhood_analysis.

    nrows : int, optional, default=4
        Number of rows in the subplot grid.

    ncols : int, optional, default=4
        Number of columns in the subplot grid.

    saveas : str, optional, default='pdf'
        Format to save the figure. Options are 'pdf' or 'png'. If 'png', the dpi is set to 300.

    showit : bool, optional, default=True
        Whether to display the plot after saving. If False, the plot will be closed after saving.

    transparent_mode : bool, optional, default=False
        Whether to save the figure with a transparent background.

    figsize : tuple of float, optional, default=(10, 7)
        Size of the figure in inches.

    Outputs:
    --------
    The function saves the ROC curves plot in the directory specified by ./nico_out/niche_prediction_linear.
    The filename will be in the format 'ROC_R<Radius>.<saveas>', where <Radius> is the radius value
    from the input and <saveas> is the file format.

    Notes:
    ------
    - The function creates a grid of ROC curves for the top 16 cell types with the highest ROC AUC values.
    """

    filename='R'+str(input.Radius)
    fig, ax = plt.subplots(nrows,ncols, figsize=figsize)
    plotaxis=[]
    for i in range(nrows):
        for j in range(ncols):
            plotaxis.append([i,j])

    highestROCofcelltype=[]
    for w in sorted(input.roc_auc, key=input.roc_auc.get, reverse=True):
        #print(w, roc_auc[w])
        highestROCofcelltype.append(w)


    for i in range(nrows*ncols):
        value=plotaxis[i]
        ax[value[0],value[1]].plot([0, 1], [0, 1], 'k--')
        ax[value[0],value[1]].set_xlim([0.0, 1.0])
        ax[value[0],value[1]].set_ylim([0.0, 1.05])
        if value[0]==(nrows-1):
            ax[value[0],value[1]].set_xlabel('False Positive Rate')
        else:
            ax[value[0],value[1]].set_xticks([])

        if i%ncols==0:
            ax[value[0],value[1]].set_ylabel('True Positive Rate')
        else:
            ax[value[0],value[1]].set_yticks([])

        ax[value[0],value[1]].set_title(str(highestROCofcelltype[i])+' : '+input.nameOfCellType[highestROCofcelltype[i]])
        ax[value[0],value[1]].plot(input.fpr[highestROCofcelltype[i]], input.tpr[highestROCofcelltype[i]], label='ROC(area = %0.2f)' % (input.roc_auc[highestROCofcelltype[i]]))

        ax[value[0],value[1]].legend(loc="best",fontsize=8)
        #ax[value[0],value[1]].grid(alpha=.4)
    snn.despine()
    #plt.suptitle('Receiver operating characteristic example')
    plt.tight_layout()
    print("The figures are saved: ",input.niche_pred_outdir+'ROC_'+filename+'.'+saveas )
    plt.savefig(input.niche_pred_outdir+'ROC_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)
    if showit:
        pass
    else:
        plt.close('all')



def read_processed_data(radius,inputdir):

    """
    Read and process the neighborhood expected feature matrix for spatial_neighborhood_analysis.

    Parameters:
    -----------
    radius : int or float
        The radius value used in the spatial analysis.

    inputdir : str
        The directory containing the input data file.

    Returns:
    --------
    neighborhoodClass : numpy.ndarray
        The matrix of neighborhood class features.

    target : numpy.ndarray
        The target labels (cell types).

    inputFeatures : range
        A range object representing the indices of the input features.

    Notes:
    ------
    - The function reads a compressed `.npz` file containing the neighborhood expected feature matrix.
    - It filters out rows with NaN values.
    - It calculates the proportion of each cell type in the dataset.
    - The function returns the processed neighborhood class features, target labels, and input feature indices.
    """

    #name=inputdir+'normalized_spatial_neighbors_'+str(radius)+'.dat'
    name=inputdir+'normalized_spatial_neighborhood_'+str(radius)+'.npz'
    data=np.load(name,allow_pickle=True)
    data1=data['input_mat_for_log_reg']
    #data1 = np.genfromtxt(open(name, "rb"), delimiter='\t', skip_header=0)
    ind=~np.isnan(data1).any(axis=1)
    data=data1[ind,:]

    prop={}
    for i in range(len(data)):
        mytype=data[i,1]
        if mytype in prop:
            prop[mytype]+=1
        else:
            prop[mytype]=1

    #print('cell type proportion')
    total=sum(prop.values())
    keys=sorted( list( prop.keys())  )
    nct=len(prop)
    featureVector=range(2,2+nct) # #just neighborhood
    neighborhoodClass= data[:,featureVector]
    target= data[:,1]
    print('data shape',data.shape, target.shape, "neighbor shape",neighborhoodClass.shape)
    inputFeatures=range(nct)
    return neighborhoodClass,target,inputFeatures



def model_log_regression(K_fold,n_repeats,neighborhoodClass,target,lambda_c,strategy,BothLinearAndCrossTerms,seed,n_jobs):
    """
    Perform logistic regression classification to learn the probabilities of each cell type class. This helper function used in spatial_neighborhood_analysis.

    Parameters:
    -----------
    K_fold : int
        Number of folds for cross-validation.

    n_repeats : int
        Number of times the cross-validation is repeated.

    neighborhoodClass : numpy.ndarray
        Matrix of neighborhood class features.

    target : numpy.ndarray
        Target labels (cell types).

    lambda_c : list or numpy.ndarray
        Regularization strength(s) to be tested in the logistic regression.

    strategy : str
        The regularization and multi-class strategy. Options include 'L1_multi', 'L1_ovr', 'L2_multi', 'L2_ovr', 'elasticnet_multi', 'elasticnet_ovr'.

    BothLinearAndCrossTerms : int
        Degree of polynomial features including interaction terms only.

    seed : int
        Random seed for reproducibility.

    n_jobs : int
        Number of jobs to run in parallel.

    Returns:
    --------
    log_reg_model : sklearn.linear_model.LogisticRegression
        The logistic regression model with specified parameters.

    parameters : dict
        Dictionary of parameters used for model training.

    hyperparameter_scoring : dict
        Dictionary of scoring metrics used for hyperparameter tuning.

    Notes:
    ------
    - The function uses polynomial features to create interaction terms based on the specified degree.
    - Hyperparameter tuning is performed using cross-validation with f1_weighted scoring metrics.
    """


    polynomial = PolynomialFeatures(degree = BothLinearAndCrossTerms, interaction_only=True, include_bias=False)


    #hyperparameter_scoring='precision_weighted'
    #hyperparameter_scoring='f1_micro'
    #hyperparameter_scoring='recall_weighted'
    hyperparameter_scoring = {#'precision_weighted': make_scorer(precision_score, average = 'weighted'),
           #'precision_macro': make_scorer(precision_score, average = 'macro'),
           #'recall_macro': make_scorer(recall_score, average = 'macro'),
           #'recall_weighted': make_scorer(recall_score, average = 'weighted'),
           #'f1_macro': make_scorer(f1_score, average = 'macro'),
            #'log_loss':'neg_log_loss',
           'f1_weighted': make_scorer(f1_score, average = 'weighted')}

    parameters = {'C':lambda_c }

    if strategy=='L1_multi':
        log_reg_model = LogisticRegression(penalty='l1',multi_class='multinomial',class_weight='balanced',solver='saga',n_jobs=n_jobs)#very slow
    if strategy=='L1_ovr':
        log_reg_model = LogisticRegression(penalty='l1',multi_class='ovr',class_weight='balanced',solver='liblinear',n_jobs=n_jobs)
    if strategy=='L2_multi':
        log_reg_model = LogisticRegression(penalty='l2',multi_class='multinomial',class_weight='balanced',solver='lbfgs',n_jobs=n_jobs)
    if strategy=='L2_ovr':
        log_reg_model = LogisticRegression(penalty='l2',multi_class='ovr',class_weight='balanced',solver='lbfgs',n_jobs=n_jobs)
    if strategy=='elasticnet_multi':
        log_reg_model = LogisticRegression(penalty='elasticnet',multi_class='multinomial',class_weight='balanced',solver='saga',n_jobs=n_jobs)
        parameters = {'C':lambda_c, 'multi_class':['ovr','multinomial'], 'l1_ratio':np.linspace(0,1,10)  }
    if strategy=='elasticnet_ovr':
        log_reg_model = LogisticRegression(penalty='elasticnet',multi_class='ovr',class_weight='balanced',solver='saga',n_jobs=n_jobs)
        parameters = {'C':lambda_c, 'multi_class':['ovr','multinomial'], 'l1_ratio':np.linspace(0,1,10)  }


    #'''
    flag=1
    how_many_times_repeat={}
    while(flag):
        seed=seed+1
        sss = RepeatedStratifiedKFold(n_splits=K_fold, n_repeats=1 ,random_state=seed)
        gs_grid = GridSearchCV(log_reg_model, parameters, scoring=hyperparameter_scoring, refit='f1_weighted',cv=sss,n_jobs=n_jobs)
        #gs_random = RandomizedSearchCV(estimator=log_reg_model, param_distributions=parameters, scoring=hyperparameter_scoring, refit='f1_weighted',cv = sss,n_jobs=n_jobs)
        pipe_grid=Pipeline([  ('polynomial_features',polynomial),   ('StandardScaler',StandardScaler()), ('logistic_regression_grid',gs_grid)])
        #pipe_random=Pipeline([  ('polynomial_features',polynomial),   ('StandardScaler',StandardScaler()), ('logistic_regression_random',gs_random)])
        pipe_grid.fit(neighborhoodClass,target)
        #pipe_random.fit(neighborhoodClass,target)

        LR_grid= pipe_grid.named_steps['logistic_regression_grid']
        lambda_c=LR_grid.best_params_['C']
        if lambda_c not in how_many_times_repeat:
            how_many_times_repeat[lambda_c]=1
        else:
            how_many_times_repeat[lambda_c]+=1

        #LR_random= pipe_random.named_steps['logistic_regression_random']
        #if LR_grid.best_params_['C']==LR_random.best_params_['C']:
        #print('Searching hyperparameters ', 'Grid method:', LR_grid.best_params_['C'], ', Randomized method:', LR_random.best_params_['C'])
        print('Searching hyperparameters ', 'Grid method:', LR_grid.best_params_['C'])
        for key in how_many_times_repeat:
            if how_many_times_repeat[key]>1:
                flag=0
                print('Inverse of lambda regularization found', lambda_c)
    #'''
    #lambda_c=0.000244140625
    #lambda_c=0.0009765625




    scorecalc=[]
    for i in range(15):
        scorecalc.append([])

    cmn=[]
    coef=[]
    seed=seed+1
    sss = RepeatedStratifiedKFold(n_splits=K_fold, n_repeats=n_repeats ,random_state=seed)

    for train_index, test_index in sss.split(neighborhoodClass,target):
        x_train,x_test=neighborhoodClass[train_index],neighborhoodClass[test_index]
        y_train,y_test=target[train_index],target[test_index]


        if strategy=='L1_multi':
            log_reg_model = LogisticRegression(C=lambda_c,penalty='l1',multi_class='multinomial',class_weight='balanced',solver='saga',n_jobs=n_jobs)#very slow
        if strategy=='L1_ovr':
            log_reg_model = LogisticRegression(C=lambda_c,penalty='l1',multi_class='ovr',class_weight='balanced',solver='liblinear',n_jobs=n_jobs)
        if strategy=='L2_multi':
            log_reg_model = LogisticRegression(C=lambda_c,penalty='l2',multi_class='multinomial',class_weight='balanced',solver='lbfgs',n_jobs=n_jobs)
        if strategy=='L2_ovr':
            log_reg_model = LogisticRegression(C=lambda_c,penalty='l2',multi_class='ovr',class_weight='balanced',solver='lbfgs',n_jobs=n_jobs)
        if strategy=='elasticnet_multi':
            log_reg_model = LogisticRegression(C=lambda_c,penalty='elasticnet',multi_class='multinomial',l1_ratio=0.5,class_weight='balanced',solver='saga',n_jobs=n_jobs)
        if strategy=='elasticnet_ovr':
            log_reg_model = LogisticRegression(C=lambda_c,penalty='elasticnet',multi_class='ovr',l1_ratio=0.5,class_weight='balanced',solver='saga',n_jobs=n_jobs)

        pipe=Pipeline([  ('polynomial_features',polynomial),   ('StandardScaler',StandardScaler()), ('logistic_regression',log_reg_model)])

        pipe.fit(x_train, y_train)
        y_pred=pipe.predict(x_test)
        y_prob = pipe.predict_proba(x_test)

        log_metric=log_loss(y_test,y_prob)
        c_k_s=cohen_kappa_score(y_test,y_pred)
        zero_met=zero_one_loss(y_test,y_pred)
        hl=hamming_loss(y_test,y_pred)
        mcc=matthews_corrcoef(y_test,y_pred)

        scorecalc[0].append(pipe.score(x_test, y_test))
        #precision, recall, fscore, support = score(y_test, predicted)
        scorecalc[1].append(f1_score(y_test, y_pred, average="macro"))
        scorecalc[2].append(precision_score(y_test, y_pred, average="macro"))
        scorecalc[3].append(recall_score(y_test, y_pred, average="macro"))
        scorecalc[4].append(f1_score(y_test, y_pred, average="micro"))
        scorecalc[5].append(precision_score(y_test, y_pred, average="micro"))
        scorecalc[6].append(recall_score(y_test, y_pred, average="micro"))
        scorecalc[7].append(f1_score(y_test, y_pred, average="weighted"))
        scorecalc[8].append(precision_score(y_test, y_pred, average="weighted"))
        scorecalc[9].append(recall_score(y_test, y_pred, average="weighted"))
        scorecalc[10].append(c_k_s)
        scorecalc[11].append(log_metric)
        scorecalc[12].append(mcc)
        scorecalc[13].append(hl)
        scorecalc[14].append(zero_met)

        poly = pipe.named_steps['polynomial_features']
        LR= pipe.named_steps['logistic_regression']
        coef.append(LR.coef_)
        cmn.append(confusion_matrix(y_test,y_pred,normalize='true'))

    cmn_std=np.std(np.array(cmn),axis=0)
    coef_std=np.std(np.array(coef),axis=0)
    comp_score_std=np.std(np.array(scorecalc),axis=1)


    cmn=np.mean(np.array(cmn),axis=0)
    coef=np.mean(np.array(coef),axis=0)
    comp_score=np.mean(np.array(scorecalc),axis=1)

    print('training',x_train.shape,'testing',x_test.shape,'coeff',coef.shape)


    #cmn=confusion_matrix(y_test,y_pred,normalize='true')
    #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes=LR.classes_.astype(int)


    #modifiedFeatures=range(1,len(CTFeatures)+1)
    fpr, tpr, roc_auc=plot_multiclass_roc(pipe, x_test, y_test, n_classes=len(classes))

    CTFeatures=poly.get_feature_names_out()
    #print("Features", CTFeatures,type(CTFeatures))

    return cmn,coef,comp_score,cmn_std,coef_std,comp_score_std,classes, lambda_c,CTFeatures,x_test,x_train,y_prob ,fpr, tpr, roc_auc



def find_interacting_cell_types(input,choose_celltypes=[],celltype_niche_interaction_cutoff=0.1,dpi=300,
coeff_cutoff=20,saveas='pdf',transparent_mode=False,showit=True,figsize=(4.0,2.0)):
    """
    Display regression coefficients indicating cell type interactions.

    Parameters:
    -----------
    input : object
        The main input is the output from spatial_neighborhood_analysis.

    choose_celltypes : list, optional
        List of cell types to display the regression coefficients for.
        If empty, the output will be shown for all cell types.
        Default is [].

    celltype_niche_interaction_cutoff : float, optional
        The cutoff value to consider for cell type niche interactions for normalized coefficients. This is visualized by blue dotted line.
        Default is 0.1.

    coeff_cutoff : int, optional
        Maximum number of neighborhood cell types shown on the X-axis of the figures for each central cell type.
        If there are too many interacting cell types, choosing a more stringet cutoff limits the display to the cell types with the largest positive or negative regression coefficients to avoid crowding in the figure.
        Default is 20.

    saveas : str, optional
        Format to save the figures in, either 'pdf' or 'png'.
        Default is 'pdf'.

    transparent_mode : bool, optional
        Background color of the figures.
        Default is False.

    showit : bool, optional
        Whether to display the figures.
        Default is True.

    figsize : tuple, optional
        Dimension of the figure size.
        Default is (4.0, 2.0).

    Outputs:
    --------
    The figures are saved in ./nico_out/niche_prediction_linear/TopCoeff_R0/*

    Notes:
    ------
    - The function normalizes the coefficients by dividing by maximum and then it visualizes by blue dotted line.
    """


    confusion_cutoff=0 # no of cell types wants to print
    filename=input.niche_pred_outdir+'TopCoeff_R'+str(input.Radius)

    nameOfCellType=input.nameOfCellType

    data=np.load(input.fout,allow_pickle=True)
    coef=data['coef']
    cmn=data['cmn']
    cmn_std=data['cmn_std']
    coef_std=data['coef_std']
    CTFeatures=data['CTFeatures']
    coef=coef/np.max(abs(coef))
    a=np.diag(cmn)
    b=np.diag(cmn_std)
    goodPredictedCellType=np.argsort(-a)
    create_directory(filename)
    #for i in range(len(goodPredictedCellType)):
    #    print(i,a[goodPredictedCellType[i]])
    # top 3 cell type in confusion matrix
    for k in range(len(a)):
        if a[goodPredictedCellType[k]]>=confusion_cutoff:
            if  nameOfCellType[goodPredictedCellType[k]] in choose_celltypes:
                flag=1
            else:
                flag=0
            if len(choose_celltypes)==0:
                flag=1

            if flag==1:
                meanCoefficients=coef[goodPredictedCellType[k]]
                stdCoefficients=coef_std[goodPredictedCellType[k]]
                highestIndex=np.argsort(-abs(meanCoefficients))

                n=min(coeff_cutoff,len(highestIndex))
                coeff_of_CT=[]
                name_of_the_coeff=[]
                std_of_coeff=[]

                #fw.write('\n'+str(k+1)+ ' Largest predicted cell type and their top 5 coefficients : '+
                #        nameOfCellType[goodPredictedCellType[k]]+' ( id = '+str(goodPredictedCellType[k])+',  confusion score = '+str('%0.2f'%a[goodPredictedCellType[k]])+')\n')

                for i in range(n):
                #for i in range(len(highestIndex)):
                    l=CTFeatures[highestIndex[i]].split()
                    temp=''
                    for j in range(len(l)):
                        temp+=nameOfCellType[int(l[j][1:])]
                        if j!=(len(l)-1):
                            temp+='--'
                    #print(temp,highestIndex[i],CTFeatures[highestIndex[i]],goodCoefficients[ highestIndex[i]   ])
                    integerName=CTFeatures[highestIndex[i]].replace('x','')
                    #fw.write(str(highestIndex[i])+'\t'+str('%0.2f'%meanCoefficients[ highestIndex[i]] ) +'\t'+temp+' ('+ integerName  +')\n')
                    coeff_of_CT.append(meanCoefficients[ highestIndex[i]])
                    name_of_the_coeff.append(temp)
                    std_of_coeff.append(stdCoefficients[ highestIndex[i]])

                fig,ax=plt.subplots( figsize=figsize)
                xx=range(len(coeff_of_CT))
                yy=np.zeros((len(coeff_of_CT)))
                zz=np.ones(len(coeff_of_CT))*celltype_niche_interaction_cutoff
                ax.errorbar(xx, coeff_of_CT, yerr=std_of_coeff,fmt='o',markeredgewidth=0,markerfacecolor=None,markeredgecolor=None,linewidth=1,capsize=1,markersize=2,elinewidth=0.1,capthick=0.75)#markeredgecolor='blue',markerfacecolor='blue',
                ax.plot(xx,yy,'k-',linewidth=0.2)
                ax.plot(xx,zz,'b:',linewidth=0.2)
                #ax.set_ylabel('value of coeff.')
                #ax.set_xlabel('name of the coeff.')
                #titlename=nameOfCellType[goodPredictedCellType[k]]+', conf score = {0:.3f}'.format(a[goodPredictedCellType[k]]) +'$\pm$'+str('%0.3f'%b[goodPredictedCellType[k]])
                titlename=nameOfCellType[goodPredictedCellType[k]]+', conf. score = {0:.3f}'.format(a[goodPredictedCellType[k]]) +'$\pm$'+str('%0.3f'%b[goodPredictedCellType[k]])

                titlename=titlename.replace('_',' ')
                ax.set_title(titlename,fontsize=7)


                ax.set_xticks(xx)
                for ind in range(len(name_of_the_coeff)):
                    name_of_the_coeff[ind]=name_of_the_coeff[ind].replace('_',' ')
                ax.set_xticklabels(name_of_the_coeff)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(90)
                    tick.set_fontsize(7)


                #fig.tight_layout()
                savefname=remove_extra_character_from_name(nameOfCellType[goodPredictedCellType[k]])
                print("The figures are saved: ", filename+'/Rank'+str(k+1)+'_'+savefname+'.'+saveas)
                fig.savefig(filename+'/Rank'+str(k+1)+'_'+savefname+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)
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


def spatial_neighborhood_analysis(
output_nico_dir=None,
anndata_object_name='nico_celltype_annotation.h5ad',
spatial_cluster_tag='nico_ct',spatial_coordinate_tag='spatial',
Radius=0,
n_repeats=1,
K_fold=5,
seed=36851234,
n_jobs=-1,
lambda_c_ranges=list(np.power(2.0, np.arange(-12, 12))),
epsilonThreshold=100,
removed_CTs_before_finding_CT_CT_interactions=[]):

    """
    Perform spatial neighborhood analysis to reconstruct the niche interaction patterns.

    This is the primary function called by the user to perform spatial neighborhood analysis, i.e., reconstruction of the niche.

    **Prerequisites:**
    Before calling this function, the user must have an annotation of the spatial cell from any method. This annotation is expected to comprise two files:
    `clusterFilename` that contains cells and cluster-ID information, and `celltypeFilename` that contains cluster-ID and cell type name information.

    **Inputs:**

    output_nico_dir : str, optional
        Directory to save the output of niche interaction prediction.
        Default is './nico_out/'.

    anndata_object_name : str, optional
        Name of the AnnData object file containing cell type annotations.
        Default is 'nico_celltype_annotation.h5ad'.

    spatial_cluster_tag : str, optional
        Slot for spatial cluster information.
        Default is 'nico_ct' that means it is stored in anndata.obs['nico_ct'] slot.

    spatial_coordinate_tag : str, optional
        Slot for spatial coordinate information.
        Default is 'spatial' that means it is stored in anndata.obsm['spatial'] slot.

    Radius : int, optional
        Niche radius to predict the cell type-cell type interactions.
        Radius 0 focuses on direct spatial neighbors inferred by Delaunay triangulation, and
        nonzero Radius extends the neighborhood to include all cells within a given radius for predicting niche interactions.
        Default is 0.

    n_repeats : int, optional
        Number of times to repeat the logistic regression after finding the hyperparameters.
        Default is 1.

    K_fold : int, optional
        Number of cross-folds for the logistic regression.
        Default is 5.

    seed : int, optional
        Random seed used in RepeatedStratifiedKFold.
        Default is 36851234.

    n_jobs : int, optional
        Number of processors to use. See https://scikit-learn.org/stable/glossary.html#term-n_jobs for details.
        Default is -1.

    lambda_c_ranges : list, optional
        The initial range of the inverse regularization parameter used in the logistic regression to find the optimal parameter.
        Default is list(np.power(2.0, np.arange(-12, 12))).

    epsilonThreshold : int, optional
        Threshold value for neighboring cell during Delaunay Triangulation. This means those cells which are large then this cutoff cannot become neighbor at any cost.
        Default is 100.

    removed_CTs_before_finding_CT_CT_interactions : list, optional
        Exclude cell types from the niche interactions analysis.
        Default is [].

    **Outputs:**

    The function saves the output of niche interaction prediction in the specified "nico_out" directory.

    **Notes:**

    - Before running this function, ensure you have the cell type annotation files in the anndata object slot.
    - If running for multiple radius parameters, it's good practice to change the output directory name or delete the previously created one.
    - If the average number of neighbors is relatively low (<1), consider increasing the radius for neighborhood analysis.
    - Every input CSV file (`positionFilename`, `clusterFilename`, `celltypeFilename`) must contain header information.

    """

    if output_nico_dir==None:
        output_nico_dir='./nico_out/'
    else:
        output_nico_dir=output_nico_dir

    BothLinearAndCrossTerms=1# If only linear terms then put 1; For both linear and crossterms use 2
    strategy='L2_multi' #Name of the strategy you want to compute the interactions options are [L1_multi, L1_ovr, L2_multi, L2_ovr, elasticnet_multi, elasticnet_ovr]
    #strategy='L1_ovr'
    #strategy='elasticnet_multi'
    removed_CTs_before_finding_CT_CT_interactions=['NM']+removed_CTs_before_finding_CT_CT_interactions

    adata=sc.read_h5ad(output_nico_dir+anndata_object_name)
    cellname=adata.obs_names.to_numpy()#df.index.to_numpy()
    cellname=np.reshape(cellname,(len(cellname),1))

    posdata=np.hstack((cellname,adata.obsm[spatial_coordinate_tag]))
    #batch=np.vstack((cellname0,adata.obs['batch'])).T

    annot = adata.obs[spatial_cluster_tag]
    ctname = sorted(list(np.unique(annot)))
    degbased_ctname=[]
    d={}
    for i in range(len(ctname)):
        degbased_ctname.append([i,ctname[i]])
        d[ctname[i]]=i
    degbased_ctname=np.array(degbased_ctname,dtype=object)
    #degbased_ctname[:,0]=degbased_ctname[:,0].astype(int)

    degbased_cluster=[]
    for i in range(len(cellname)):
        degbased_cluster.append([  adata.obs_names[i],d[annot[i]] ])
    degbased_cluster=np.array(degbased_cluster,dtype=object)


    if BothLinearAndCrossTerms==1:
        niche_pred_outdir=output_nico_dir+'niche_prediction_linear/'
    else:
        niche_pred_outdir=output_nico_dir+'niche_prediction_cross/'
    create_directory(niche_pred_outdir)


    PP,cluster,noct,fraction_CT,clusterWithBarcodeId= reading_data(posdata,degbased_cluster,degbased_ctname,output_nico_dir,removed_CTs_before_finding_CT_CT_interactions)
    M, neighbors,distance=create_spatial_CT_feature_matrix(Radius,PP,cluster,noct,fraction_CT,niche_pred_outdir,epsilonThreshold)
    pickle.dump(neighbors,open(output_nico_dir+'neighbors_'+str(Radius)+'.p', 'wb'))
    pickle.dump(distance,open(output_nico_dir+'distances_'+str(Radius)+'.p','wb'))
    df=pd.DataFrame(clusterWithBarcodeId)
    df.to_csv(output_nico_dir+'used_Clusters'+str(Radius)+'.csv',index=False)

    f=open(output_nico_dir+'used_CT.txt')
    nameOfCellType={}
    for line in f:
        l=line[0:-1].split('\t')
        nameOfCellType[int(l[0])]=l[1]
    f.close()



    #fw=open(mainoutputdir+'prediction_R'+str(Radius)+'.dat','w')
    #fw.write('\nRadius = '+ str(Radius)  + '\n')
    fout=niche_pred_outdir+'classifier_matrices_'+str(Radius)+'.npz'

    inputdata={}
    inputdata['outputdir']=output_nico_dir
    inputdata['fout']=fout
    inputdata['niche_pred_outdir']=niche_pred_outdir
    inputdata['nameOfCellType']=nameOfCellType
    inputdata['Radius']=Radius
    inputdata['BothLinearAndCrossTerms']=BothLinearAndCrossTerms


    neighborhoodClass,target,inputFeatures=read_processed_data(Radius,niche_pred_outdir)
    cmn,coef,comp_score,cmn_std,coef_std,comp_score_std,classes,lambda_c,CTFeatures,x_test,x_train,predicted_probs,fpr, tpr, roc_auc=model_log_regression(K_fold, n_repeats,neighborhoodClass,target,lambda_c_ranges,strategy,BothLinearAndCrossTerms,seed,n_jobs)
    score=np.array([comp_score, comp_score_std]).T
    np.savez(fout,cmn=cmn,coef=coef,cmn_std=cmn_std,coef_std=coef_std,CTFeatures=CTFeatures)

    inputdata['classes']=classes
    inputdata['lambda_c']=lambda_c
    inputdata['fpr']=fpr
    inputdata['tpr']=tpr
    inputdata['roc_auc']=roc_auc
    inputdata['x_test']=x_test
    inputdata['x_train']=x_train
    inputdata['predicted_probs']=predicted_probs
    inputdata['inputFeatures']=inputFeatures
    inputdata['score']=score
    output=SimpleNamespace(**inputdata)

    return output











def plot_evaluation_scores(input,saveas='pdf',transparent_mode=False,showit=True,dpi=300,figsize=(4,3)):

    """
    This function generates and saves plots of evaluation scores obtained from the spatial_neighborhood_analysis.
    The plots can be saved in PDF or PNG format and can be displayed during execution.

    Parameters
    ----------
    input : dict or similar
        The main input is the output from spatial_neighborhood_analysis. This should contain the evaluation scores to be plotted.
    saveas : str, optional
        Format to save the figures. Options are 'pdf' or 'png'. Default is 'pdf'.
    transparent_mode : bool, optional
        If True, the background color of the figures will be transparent. Default is False.
    showit : bool, optional
        If True, the figures will be displayed when the function is called. Default is True.
    figsize : tuple, optional
        Dimensions of the figure size in inches (width, height). Default is (4, 3).

    Outputs
    -------
    None
        The function saves the generated figures in the directory "./nico_out/niche_prediction_linear/" with filenames starting with "scores".

    Notes
    -----
    - The order of scores saved in input.score as follows:

        - 1. accuracy
        - 2. macro F1
        - 3. macro precision
        - 4. macro recall
        - 5. micro F1
        - 6. micro precision
        - 7. micro recall
        - 8. weighted F1
        - 9. weighted precision
        - 10. weighted recall
        - 11. Cohen Kappa
        - 12. cross entropy
        - 13. mathhew correlation coefficient
        - 14. heming loss
        - 15. zeros one loss

    """


    xlabels=['accuracy','macro F1','macro precision','macro recall','micro [all]','weighted F1','weighted precision','weighted recall','cohen kappa','mcc']
    index=[0,1,2,3,4,7,8,9,10,12]

    data=input.score

    fig,axs=plt.subplots(1,1,figsize=figsize)
    #name=mainoutputdir+'matrix_score_R'+str(Radius)+'.dat'
    #data=np.genfromtxt(open(name, "rb"), delimiter=',', skip_header=0)
    yt=data[index,0]
    xt=range(len(yt))
    axs.bar(xt,yt)#,color='b')
    lowRg=yt-data[index,1]
    highRg=yt+data[index,1]
    axs.errorbar(xt,yt,yerr=data[index,1],fmt='o',color='k',ms=0.5,lw=0.2)
    #axs.fill_between(xt, lowRg, highRg,facecolor='b',alpha=0.2)
    #legend1= axs.legend(loc='lower left',bbox_to_anchor=(0.02, 0.05),ncol=1, borderaxespad=0., prop={"size":6},fancybox=True, shadow=True)

    axs.set_xticks(range(len(index)))
    ytstd=max(data[index,1])
    axs.set_yticks(np.linspace(min(yt)-ytstd,max(yt)+ytstd,4))

    axs.set_xticklabels(xlabels)
    for tick in axs.get_xticklabels():
        tick.set_rotation(90)

    axs.set_ylabel('score')
    fig.tight_layout()

    print("The figures are saved: ", input.niche_pred_outdir+'scores_'+str(input.Radius)+'.'+saveas)
    fig.savefig(input.niche_pred_outdir+'scores_'+str(input.Radius)+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)
    if showit:
        pass
    else:
        plt.close('all')




def plot_niche_interactions_without_edge_weight(input,niche_cutoff=0.1,
saveas='pdf',transparent_mode=False,showit=True,figsize=(10,7),dpi=300,input_colormap='jet',
with_labels=True,node_size=300,linewidths=0.5, node_font_size=8, alpha=0.5,font_weight='normal'):

    """
    Plot niche interactions map without edge weights.

    This function generates and saves a niche interactions map using data from the output of spatial_neighborhood_analysis.
    The graph illustrates connections between cell types based on their niche interactions, without weighting the edges
    by interaction strength. The output plot can be saved in either PDF or PNG format, and optionally displayed after generation.

    Parameters
    ----------
    input : dict or similar
        The main input containing data from spatial_neighborhood_analysis. This should include information on cell types
        and interaction strengths needed to plot niche cell type interactions.
    niche_cutoff : float, optional
         Threshold for plotting connections in the niche interactions map. Only connections with normalized interaction
        strengths above this cutoff are displayed. Higher values reduce connections, while lower values increase them.
        Default is 0.1.
    saveas : str, optional
        Format to save the figures. Options are 'pdf' or 'png'. Default is 'pdf'.
    transparent_mode : bool, optional
        If True, the plot background will be transparent. Default is False.
    showit : bool, optional
        If True, the figures will be displayed when the function is called. Default is True.
    figsize : tuple, optional
        Dimensions of the plot in inches (width, height). Default is (10, 7).
    dpi : int, optional
        Resolution in dots per inch for saving the figure. Default is 300.
    input_colormap : str, optional
        Color map for node colors, based on matplotlib colormaps. Default is 'jet'.
        For details see documentation https://matplotlib.org/stable/gallery/color/colormap_reference.html
    with_labels : bool, optional
        If True, displays cell type labels on the nodes. Default is True.
    node_size : int, optional
        Size of the nodes. Default is 300.
    linewidths : int, optional
        Width of the node border lines. Default is 0.5.
    node_font_size : int, optional
        Font size for node labels. Default is 8.
    alpha : float, optional
        Opacity level for nodes and edges. Default is 0.5.
    font_weight : str, optional
        Weight of the font for node labels. Options are 'normal' or 'bold'. Default is 'normal'.


    Outputs
    -------
    None
        The function saves the generated figures in the directory "./nico_out/niche_prediction_linear/" with filenames starting with "Niche_interactions_*".
    """

    data=np.load(input.fout,allow_pickle=True)
    coef=data['coef']
    cmn=data['cmn']
    cmn_std=data['cmn_std']
    coef_std=data['coef_std']
    CTFeatures=data['CTFeatures']

    n=len(input.nameOfCellType)
    top=cm.get_cmap(input_colormap)
    cumsum=np.linspace(0,1,n)
    newcmp=ListedColormap(top(cumsum))
    #newcmp=ListedColormap(top(np.linspace(0.95,0.15,n)),name="OrangeBlue")
    gradientColor=newcmp.colors
    #print(len(gradientColor),len(top))
    G = nx.DiGraph()
    #coef=coef[0:3,0:19]

    labeldict={}
    for i in range(len(input.nameOfCellType)):
        labeldict[i]=input.nameOfCellType[i]
        G.add_node(i,color=gradientColor[i])

    #fig,ax=plt.subplots( figsize=figsize)
    edges=[]
    #print('coef',coef.shape, np.max(abs(coef)))
    norm_coef=coef/np.max(abs(coef))

    largest=[]
    for i in range(coef.shape[0]):
        for j in range(coef.shape[1]):
            if norm_coef[i,j]>niche_cutoff:
                if i!=j:
                    #edges.append([i,j,coef[i,j]])
                    largest.append(norm_coef[i,j])
                    G.add_edge(j,i,color=gradientColor[i],weight=norm_coef[i,j])

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    edge_labels = dict([((n1, n2), '%0.2f'%G[n1][n2]['weight']) for n1, n2 in G.edges])

    pos = nx.nx_pydot.graphviz_layout(G)

    fig,ax=plt.subplots( figsize=figsize)
    nx.draw(G, pos = pos,labels=labeldict,with_labels=with_labels,node_size=node_size,linewidths=linewidths, font_size=node_font_size, font_weight=font_weight,width=weights, alpha=alpha,edge_color=colors,node_color=gradientColor)
    #labels=nx.draw_networkx_labels(G,pos=pos,**options)
    #nx.draw(G,pos=pos,**options)
    #nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels,label_pos=0.35, font_size=3  )
    print("The figures are saved: ", input.niche_pred_outdir+'Niche_interactions_without_edge_weights_R'+str(input.Radius) +'.'+saveas)
    fig.savefig(input.niche_pred_outdir+'Niche_interactions_without_edge_weights_R'+str(input.Radius) +'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)
    if showit:
        pass
    else:
        plt.close('all')



def plot_niche_interactions_with_edge_weight(input,niche_cutoff=0.1,saveas='pdf',transparent_mode=False,showit=True,figsize=(10,7),
dpi=300,input_colormap='jet',with_labels=True,node_size=300,linewidths=0.5, node_font_size=8, alpha=0.5,font_weight='normal',
edge_label_pos=0.35,edge_font_size=3):
        #niche_cutoff it is normalized large value has fewer connections and small value has larger connections
        """
        Plot niche interactions map with edge weights.

        This function generates and saves a directed graph that represents niche interactions map based on the output of spatial_neighborhood_analysis.
        The nodes represent cell types, and the edges (with weights) indicate the strength of interactions between central cell typ and niche cell types.
        The plot can be saved in PDF or PNG format and can be displayed during execution.

        Parameters
        ----------
        input : dict or similar
            The main input is the output from spatial_neighborhood_analysis. This should contain the necessary data to plot the niche interactions.
        niche_cutoff : float, optional
            Threshold for including interactions in the graph. Higher values result in fewer connections, while lower values include more connections. Default is 0.1.
        saveas : str, optional
            The format for saving the figures. Options are 'pdf' or 'png'. Default is 'pdf'.
        transparent_mode : bool, optional
            If True, saves the figure with a transparent background. Default is False.
        showit : bool, optional
            If True, displays the plot after generating. Default is True.
        figsize : tuple, optional
            Size of the figure in inches (width, height). Default is (10, 7).
        dpi : int, optional
            Resolution in dots per inch for saving the figure. Default is 300.
        input_colormap : str, optional
            Color map for node colors, based on matplotlib colormaps. Default is 'jet'.
            For details see documentation https://matplotlib.org/stable/gallery/color/colormap_reference.html
        with_labels : bool, optional
            If True, displays cell type labels on the nodes. Default is True.
        node_size : int, optional
            Size of the nodes. Default is 300.
        linewidths : int, optional
            Width of the node border lines. Default is 0.5.
        node_font_size : int, optional
            Font size for node labels. Default is 8.
        alpha : float, optional
            Opacity level for nodes and edges. Default is 0.5.
        font_weight : str, optional
            Weight of the font for node labels. Options are 'normal' or 'bold'. Default is 'normal'.
        edge_label_pos : float, optional
            Position of edge labels along the edges. Default is 0.35.
        edge_font_size : int, optional
            Font size for edge labels. Default is 3.

        Outputs
        -------
        None
            The function saves the generated figures in the directory "./nico_out/niche_prediction_linear/" with filenames starting with "Niche_interactions_*".

        """

        data=np.load(input.fout,allow_pickle=True)
        coef=data['coef']
        cmn=data['cmn']
        cmn_std=data['cmn_std']
        coef_std=data['coef_std']
        CTFeatures=data['CTFeatures']

        #print(coef.shape,len(CTFeatures))


        n=len(input.nameOfCellType)
        top=cm.get_cmap(input_colormap)
        cumsum=np.linspace(0,1,n)
        newcmp=ListedColormap(top(cumsum))
        #newcmp=ListedColormap(top(np.linspace(0.95,0.15,n)),name="OrangeBlue")
        gradientColor=newcmp.colors
        #print(len(gradientColor),len(top))
        G = nx.DiGraph()
        #coef=coef[0:3,0:19]

        labeldict={}
        for i in range(len(input.nameOfCellType)):
            labeldict[i]=input.nameOfCellType[i]
            G.add_node(i,color=gradientColor[i])

        fig,ax=plt.subplots( figsize=figsize)
        edges=[]
        #print('coef',coef.shape, np.max(abs(coef)))
        norm_coef=coef/np.max(abs(coef))

        largest=[]
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                if norm_coef[i,j]>niche_cutoff:
                    if i!=j:
                        #edges.append([i,j,coef[i,j]])
                        largest.append(norm_coef[i,j])
                        G.add_edge(j,i,color=gradientColor[i],weight=norm_coef[i,j])
                        #print(i,nameOfCellType[i],nameOfCellType[j],coef[i,j]
        #G.add_edges_from([(1, 2,10), (1, 3,20)])
        #G.add_weighted_edges_from(edges)


        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        #ncolor=[G[u]['color'] for u in G.nodes()]

        #edge_labels = dict([((n1, n2), f'{n1}->{n2}') for n1, n2 in G.edges])
        edge_labels = dict([((n1, n2), '%0.2f'%G[n1][n2]['weight']) for n1, n2 in G.edges])

        pos = nx.nx_pydot.graphviz_layout(G)


        #pos=nx.spring_layout(G)
        #pos=nx.circular_layout(G)
        nx.draw(G, pos = pos,
        labels=labeldict,with_labels=with_labels,node_size=node_size,
         linewidths=linewidths, font_size=node_font_size, font_weight=font_weight,width=weights, alpha=alpha,edge_color=colors,node_color=gradientColor)

        nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels,label_pos=edge_label_pos, font_size=edge_font_size)
        print("The figures are saved: ", input.niche_pred_outdir+'Niche_interactions_with_edge_weights_R'+str(input.Radius)+'.'+saveas)
        fig.savefig(input.niche_pred_outdir+'Niche_interactions_with_edge_weights_R'+str(input.Radius)+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=dpi)
        if showit:
            pass
        else:
            plt.close('all')
