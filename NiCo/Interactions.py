

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
    "This function creates empty directory."
    answer=os.path.isdir(outputFolder)
    if answer==True:
        pass
    else:
        os.mkdir(outputFolder)

def findNeighbors_in_given_radius(location,radius):
    """
    The helper function is used in create_spatial_CT_feature_matrix to find the neighbors for each cell using the radius parameter.
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
    The helper function is used in create_spatial_CT_feature_matrix to find the neighbors for each cell using the Delaunay triangulation.
    """
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]


def create_spatial_CT_feature_matrix(radius,PP,louvain,noct,fraction_CT,saveSpatial,epsilonThreshold):
    """
    Helper function used in spatial_neighborhood_analysis to generate the expected spatial cell type neighborhood matrix.
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

    '''
    expectedNeighbors=np.array(expectedNeighbors)
    M=[]
    for i in range(len(noct)):
        a=np.where(expectedNeighbors[:,0]==i)
        b=np.where(expectedNeighbors[:,0]!=i)
        #print('a',len(a[0]),len(b[0]))
        myCT=np.mean(expectedNeighbors[a[0],1:],axis=0)
        remainCT=np.mean(expectedNeighbors[b[0],1:],axis=0)
        M.append(myCT/remainCT)
        #print(i,M[i])
    M=np.array(M)
    '''
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

    '''
    CTname=[]
    with open(celltypeFilename,'r') as f:
        cont = f.read()
        lines=cont.split('\n')
        CTname=[]
        CTid=[]
        for i in range(1,len(lines)):
            l=lines[i].split(delimiter)
            if len(l)>1:
                name=l[1]
                '
                name=l[1].replace('/','_')
                name=name.replace(' ','_')
                name=name.replace('"','')
                name=name.replace("'",'')
                name=name.replace(')','')
                name=name.replace('(','')
                name=name.replace('+','p')
                name=name.replace('-','n')
                name=name.replace('.','')
                '
                if name not in removed_CTs_before_finding_CT_CT_interactions:
                    CTname.append(name)
                    CTid.append(int(l[0]))
    '''

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
    Compute the ROC for each cell type prediction and its performance on the test dataset.
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


def plot_confusion_matrix(input,saveas='pdf',showit=True,transparent_mode=False,figsize=(5.5,5)):
    """
    Inputs:

    The primary input is the output from spatial_neighborhood_analysis.

    | Save the figures in PDF or PNG format (dpi for PNG format is 300)
    | (default) saveas='pdf'

    | Background color in the figures
    | (default) transparent_mode=False

    | Dimension of the figure size.
    | (default) figsize=(5.5,5),

    Outputs:

    | The figures are saved in ./spatial_ct_ct_interactions/niche_prediction_linear/*
    | Confusing_matrix_* shows the confusion matrix of the logistic regression classifier.

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
    plt.savefig(input.niche_pred_outdir+'Confusing_matrix_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
    if showit:
        pass
    else:
        plt.close('all')

def plot_coefficient_matrix(input,saveas='pdf',showit=True,transparent_mode=False,figsize=(5,8)):
    """
    Inputs:

    The primary input is the output from spatial_neighborhood_analysis.

    | Save the figures in PDF or PNG format (dpi for PNG format is 300)
    | (default) saveas='pdf'

    | Background color in the figures
    | (default) transparent_mode=False

    | Dimension of the figure size.
    | (default) figsize=(5,8)

    Outputs:

    | The figures are saved in ./spatial_ct_ct_interactions/niche_prediction_linear/*
    | weight_matrix_* shows the learned coefficient matrix from the logistic regression classifier.

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
    plt.savefig(input.niche_pred_outdir+'weight_matrix_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
    if showit:
        pass
    else:
        plt.close('all')


def plot_predicted_probabilities(input,saveas='pdf',showit=True,transparent_mode=False,figsize=(12,6)):
    """
    Inputs:

    The primary input is the output from spatial_neighborhood_analysis.

    | Save the figures in PDF or PNG format (dpi for PNG format is 300)
    | (default) saveas='pdf'

    | Background color in the figures
    | (default) transparent_mode=False

    | Dimension of the figure size.
    | (default) figsize=(12,6)

    Outputs:

    | The figures are saved in ./spatial_ct_ct_interactions/niche_prediction_linear/*
    | predicted_probability_* shows the training, testing, and predicted probability used in the logistic regression classifier.

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
    plt.savefig(input.niche_pred_outdir+'predicted_probability_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)

    if showit:
        pass
    else:
        plt.close('all')
    #print(predicted_probs)
    #prob=sigmoid( np.dot([y_train, y_test,1], log_reg_model.coef_.T) + log_reg_model.intercept_ )
    #print(prob)


def plot_roc_results(input,nrows=4,ncols=4,saveas='pdf',showit=True,transparent_mode=False,figsize=(10, 7),):

    """
    Inputs:

    The primary input is the output from spatial_neighborhood_analysis.

    | The top 16 cell types predictions are shown as an ROC curve. These 16 are plotted as four-by-four subplots
    | (default) nrows=4, ncols=4

    | Save the figures in PDF or PNG format (dpi for PNG format is 300)
    | (default) saveas='pdf'

    | Background color in the figures
    | (default) transparent_mode=False

    | Dimension of the figure size.
    | (default) figsize=(10, 7)

    Outputs:

    | The figures are saved in ./spatial_ct_ct_interactions/niche_prediction_linear/*
    | ROC_* shows the Receiver operating characteristic for the top 16 cell types.

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
    plt.savefig(input.niche_pred_outdir+'ROC_'+filename+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
    if showit:
        pass
    else:
        plt.close('all')
    #'''









def read_processed_data(radius,inputdir):
    """
    Helper function used in spatial_neighborhood_analysis to read the neighborhood expected feature matrix.
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
    The helper function used in spatial_neighborhood_analysis to perform the logistic regression classifier to learn the probabilities of each cell type class.
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
    '''
    print("accuracy score\t",np.mean(scorecalc[0]))
    print("\nmacro")
    print("f1 score\t",np.mean(scorecalc[1]))
    print("precision score\t",np.mean(scorecalc[2]))
    print("recall score\t",np.mean(scorecalc[3]))

    print("\nmicro f1, precision, recall all same")
    print("score\t",np.mean(scorecalc[4]))
    #print("precision score in 10 run\t",np.mean(scorecalc[5]))
    #print("recall score in 10 run\t",np.mean(scorecalc[6]))

    print("\nWeighted")
    print("f1 score\t",np.mean(scorecalc[7]))
    print("precision\t",np.mean(scorecalc[8]))
    print("recall score\t",np.mean(scorecalc[9]))


    print('\ncohen_kappa_score (best=1): {0:.4f}'.format(np.mean(scorecalc[10])))
    print('log_loss or cross entropy (best=lowest): {0:.4f}'.format(np.mean(scorecalc[11])))
    print('matthews_corrcoef: {0:.4f}'.format( np.mean(scorecalc[12])  ))
    print('hemming_loss (best=lowest): {0:.4f}'.format( np.mean(scorecalc[13] )))
    print('zero_one_loss (best=0): {0:.4f}'.format(np.mean(scorecalc[14])))
    '''

    return cmn,coef,comp_score,cmn_std,coef_std,comp_score_std,classes, lambda_c,CTFeatures,x_test,x_train,y_prob ,fpr, tpr, roc_auc



def find_interacting_cell_types(input,choose_celltypes=[],celltype_niche_interaction_cutoff=0.1,
coeff_cutoff=20,saveas='pdf',transparent_mode=False,showit=True,figsize=(4.0,2.0)):
    """
    Inputs:

    The main input is the output from spatial_neighborhood_analysis.

    | The cell type for which you would like to display the the regression coefficients indicating cell type interaction
    | (default) choose_celltypes=[]
    | If the list is empty, then the output will be shown for all cell types.

    | The maximum number of neighborhood cell types shown on the X-axis of the figures for each central cell type
    | If there are too many interacting cell types, choosing a more stringet cutoff limits the display to the cell types with the largest positive or negative regression coefficients to avoid crowding of the figure
    | (default) coeff_cutoff=20

    | Save the figures in PDF or PNG format (dpi for PNG format is 300)
    | (default) saveas='pdf'

    | Background color in the figures
    | (default) transparent_mode=False

    | Dimension of the figure size.
    | (default) figsize=(4,2)

    Outputs:

    | The figures are saved in ./spatial_ct_ct_interactions/niche_prediction_linear/TopCoeff_R0/*

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
                fig.savefig(filename+'/Rank'+str(k+1)+'_'+savefname+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
                if showit:
                    pass
                else:
                    plt.close('all')

'''
def plot_celltype_nich_prediction_from_neighborhood_all_together(cmn,coef,cmn_std,coef_std,figuresize,filename):
            a=np.diag(cmn)
            b=np.diag(cmn_std)
            goodPredictedCellType=np.argsort(-a)
            create_directory(filename)
            fig,ax=plt.subplots( figsize=(figuresize[0],figuresize[1]))
            xx=range(len(coeff_of_CT))
            yy=np.zeros((len(coeff_of_CT)))
            ax.errorbar(xx, coeff_of_CT, yerr=std_of_coeff,fmt='o',markeredgewidth=0,markerfacecolor=None,markeredgecolor=None,linewidth=1,capsize=1,markersize=2,elinewidth=0.1,capthick=0.75)#markeredgecolor='blue',markerfacecolor='blue',
            ax.plot(xx,yy,'k-',linewidth=0.2)
            #ax.set_ylabel('value of coeff.')
            #ax.set_xlabel('name of the coeff.')
            #titlename=nameOfCellType[goodPredictedCellType[k]]+', conf score = {0:.3f}'.format(a[goodPredictedCellType[k]]) +'$\pm$'+str('%0.3f'%b[goodPredictedCellType[k]])
            titlename=nameOfCellType[goodPredictedCellType[k]]+', id = '+str(goodPredictedCellType[k])+', conf score = {0:.3f}'.format(a[goodPredictedCellType[k]]) +'$\pm$'+str('%0.3f'%b[goodPredictedCellType[k]])

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
            fig.savefig(filename+'/Rank'+str(k+1)+'_'+savefname+'.pdf',bbox_inches='tight',transparent=True)
            fig.savefig(filename+'/Rank'+str(k+1)+'_'+savefname,bbox_inches='tight',transparent=True,dpi=300)
            fig.clf()
'''

def remove_extra_character_from_name(name):
    """
    This function removes special characters from the cell type names to avoid throwing an error while saving the figures.
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


        '''
        **This is the primary function called by the user to perform spatial neighborhood analysis, i.e., reconstruction of the niche.**

        **Before calling this function, the user must have an annotation of the spatial cell from any method. It can be obtained by NiCo's annotation module.
        This annotation is expected to comprise two files, clusterFilename that contain cells and cluster-ID information,
        and celltypeFilename that contains cluster-ID and cell type name information.**

        call spatial_neighborhood_analysis function from the interaction module.
        Perform spatial neighborhood analysis to find the niche interaction patterns.

        If the user wants to run for multiple radius parameters, then it is good practice to change the outputdir directory name or delete the previously created one.

        If the average number of neighbors is relatively low (<1), increase the radius to perform neighborhood analysis.

        Every input CSV file (positionFilename,clusterFilename,celltypeFilename) must contain the header information.

        Inputs:

        | Please change the following two filenames (an paths) if you are using external cell type annotation
        | The cluster partition filename from NiCo or any standard clustering method (cell barcode, cluster-ID)
        | (default) clusterFilename='./inputQuery/MNN_based_annotations/3_deg_annotation_spatial_cluster.csv'

        | The cell type information filename from NiCo or any standard clustering method (cluster-ID, cell type name)
        | (default) celltypeFilename='./inputQuery/MNN_based_annotations/3_deg_annotation_spatial_ct_name.dat'



        | The delimiter used in the celltypeFilename and clusterFilename
        | These files should have the header information
        | delimiter=','

        | Niche radius to predict the cell type-cell type interactions
        | Radius 0 focuses on direct spatial neighbors infered by Delaunay triangulation, and
        | nonzero Radius extends the neighborhood to include all cells within a given radius for predicting niche interactions.
        | (default) Radius=0

        | Number of times to run the logistic regression after finding the hyperparameters
        | (default) n_repeats=1

        | Number of cross-folds
        | (default) K_fold=5

        | Random seed used in RepeatedStratifiedKFold
        | seed=36851234

        | Number of used processors For details, see here https://scikit-learn.org/stable/glossary.html#term-n_jobs
        | n_jobs=-1

        | The initial range of the inverse regularization parameter used in the logistic regression to find the optimal parameter
        | (default) lambda_c_ranges=list(np.power(2.0, np.arange(-12, 12)))

        | Exclude cell types from the niche interactions analysis
        | (default) removed_CTs_before_finding_CT_CT_interactions=[]

        Outputs:

        | The output of niche interaction prediction
        | (default) output_niche_prediction_dir='./spatial_ct_ct_interactions/'


        '''

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







def plot_evaluation_scores(input,saveas='pdf',transparent_mode=False,showit=True,figsize=(4,3)):

    """
    Inputs:

    The main input is the output from spatial_neighborhood_analysis.

    | Save the figures in PDF or PNG format (dpi for PNG format is 300)
    | (default) saveas='pdf'

    | Background color in the figures
    | (default) transparent_mode=False

    | Dimension of the figure size.
    | (default) figsize=(4,3)

    Outputs:

    | The figures are saved in "./spatial_ct_ct_interactions/niche_prediction_linear/scores*"

    """

    ## The order of all 15 scores are following
    #1-4 'accuracy','macro F1','macro precision','macro recall',
    #5-7 'micro [all]',
    #8-11 'weighted F1','weighted precision','weighted recall','cohen kappa',
    #12=15 'cross entropy', 'matthew correlation coefficient','heming loss', 'zero one loss'

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
    fig.savefig(input.niche_pred_outdir+'scores_'+str(input.Radius)+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
    if showit:
        pass
    else:
        plt.close('all')


'''
def plot_normalized_coefficients_radius_wise(inputRadius,BothLinearAndCrossTerms,inputdir,strategy,figuresize):
    f=open(inputdir+'BiologicalNameOfCT.dat')
    nameOfCellType={}
    featureName=[]
    for line in f:
        l=line[0:-1].split('\t')
        nameOfCellType[int(l[0])]=l[1]
        featureName.append(l[1])
    savedir=mainoutputdir+"RadiusWiseNormalizedCoefficients/"
    create_directory(savedir)

    coef=[]
    confusion=[]
    for radius in inputRadius:
        fname=mainoutputdir+'classifier_matrices_'+str(radius)+'.npz'
        data=np.load(fname,allow_pickle=True)
        coef.append(data['coef'])
        cmn=data['cmn']
        #cmn_std=data['cmn_std']
        #coef_std=data['coef_std']
        CTFeatures=data['CTFeatures']
        #coef.append(np.loadtxt(maindir+'matrix_avg_coefficients_R'+str(radius)+'.dat',delimiter=','))
        #cmn=np.loadtxt(maindir+'matrix_avg_confusion_R'+str(radius)+'.dat',delimiter=',')
        confusion.append(np.diagonal(cmn))


    coefSize=coef[0].shape
    coef=np.array(coef)
    C=np.array(confusion)
    B=np.einsum('kij->ikj',coef)
    #print(B.shape,C.shape)


    name_of_the_coeff=[]
    n=coefSize[1]
    for i in range(n):
        l=CTFeatures[i].split()
        temp=''
        for j in range(len(l)):
            temp+=nameOfCellType[int(l[j][1:])]
            if j!=(len(l)-1):
                temp+='--'
        name_of_the_coeff.append(temp)


    for i in range(len(B)):
        fig,ax=plt.subplots(1,1,figsize=(figuresize[0],figuresize[1]))
        for j in range(len(B[i])):
            value=np.max(abs(B[i][j]))
            B[i][j]=B[i][j]/value
        snn.heatmap(B[i],xticklabels=name_of_the_coeff)#,xticklabels=classes)
        ax.set_yticklabels(inputRadius,fontsize=5, rotation=0)
        ax.set_xticklabels(name_of_the_coeff,fontsize=5, rotation=90)
        ax.set_title(featureName[i] + ' [%0.2f'%C[0,i] + ', %0.2f]'%C[-1,i]   ,fontsize=7)
        fig.tight_layout()
        fig.savefig(savedir+'CT_'+str(i+1)+ '_'+featureName[i]  + '.png',bbox_inches='tight',dpi=300)
        fig.clf()
'''

def plot_niche_interactions_without_edge_weight(input,niche_cutoff=0.1,saveas='pdf',transparent_mode=False,showit=True,figsize=(10,7)):

    """
    Inputs:

    The main input is the output from spatial_neighborhood_analysis.

    | Cutoff used for plotting the niche interactions map
    | (default) niche_cutoff=0.10

    | Save the figures in PDF or PNG format (dpi for PNG format is 300)
    | (default) saveas='pdf'

    | Background color in the figures
    | (default) transparent_mode=False

    | Dimension of the figure size.
    | (default) figsize=(10,7)

    Outputs:

    | The figures are saved in ./spatial_ct_ct_interactions/niche_prediction_linear/Niche_interactions_*

    """

    data=np.load(input.fout,allow_pickle=True)
    coef=data['coef']
    cmn=data['cmn']
    cmn_std=data['cmn_std']
    coef_std=data['coef_std']
    CTFeatures=data['CTFeatures']

    n=len(input.nameOfCellType)
    top=cm.get_cmap('jet')
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
    nx.draw(G, pos = pos,labels=labeldict,with_labels=True,node_size=300,linewidths=1, font_size=8, width=weights, edge_color=colors,node_color=gradientColor)
    #labels=nx.draw_networkx_labels(G,pos=pos,**options)
    #nx.draw(G,pos=pos,**options)
    #nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels,label_pos=0.35, font_size=3  )
    print("The figures are saved: ", input.niche_pred_outdir+'Niche_interactions_without_edge_weights_R'+str(input.Radius) +'.'+saveas)
    fig.savefig(input.niche_pred_outdir+'Niche_interactions_without_edge_weights_R'+str(input.Radius) +'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
    if showit:
        pass
    else:
        plt.close('all')



def plot_niche_interactions_with_edge_weight(input,niche_cutoff=0.1,saveas='pdf',transparent_mode=False,showit=True,figsize=(10,7)):
        #niche_cutoff it is normalized large value has fewer connections and small value has larger connections
        """
        Inputs:

        The main input is the output from spatial_neighborhood_analysis.

        | Cutoff used for plotting the niche interactions map
        | (default) niche_cutoff=0.10

        | Save the figures in PDF or PNG format (dpi for PNG format is 300)
        | (default) saveas='pdf'

        | Background color in the figures
        | (default) transparent_mode=False

        | Dimension of the figure size.
        | (default) figsize=(10,7)

        Outputs:

        | The figures are saved in ./spatial_ct_ct_interactions/niche_prediction_linear/Niche_interactions_*

        """



        data=np.load(input.fout,allow_pickle=True)
        coef=data['coef']
        cmn=data['cmn']
        cmn_std=data['cmn_std']
        coef_std=data['coef_std']
        CTFeatures=data['CTFeatures']

        #print(coef.shape,len(CTFeatures))

        '''
        meanCoefficients=coef[0]
        stdCoefficients=coef_std[0]
        #highestIndex=np.argsort(-abs(meanCoefficients))

        n=min(len(meanCoefficients))
        coeff_of_CT=[]
        name_of_the_coeff=[]
        std_of_coeff=[]


        for i in range(n):
            l=CTFeatures[i].split()
            temp=''
            for j in range(len(l)):
                temp+=nameOfCellType[int(l[j][1:])]
                if j!=(len(l)-1):
                    temp+='--'
        integerName=CTFeatures[i].replace('x','')
        coeff_of_CT.append(meanCoefficients[i])
        name_of_the_coeff.append(temp)
        std_of_coeff.append(stdCoefficients[i])
        #sklearn.cluster.bicluster
        linked=linkage(coef,'single')
        #plt.figure(figsize=(10, 7))
        fig,ax=plt.subplots( figsize=(figsize_niche[0],figsize_niche[1]))
        dendrogram(linked,
            orientation='top',
            labels=nameOfCellType,
            distance_sort='descending',
            show_leaf_counts=True)
        #ax.set_xticks(xx)
        #ax.set_xticklabels(name_of_the_coeff)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        fig.savefig(mainoutputdir+'HierarchicalClustering_Rad'+str(Radius),bbox_inches='tight',dpi=300)
        plt.close('all')
        '''

        n=len(input.nameOfCellType)
        top=cm.get_cmap('jet')
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


        '''
        #layout
        pos=nx.fruchterman_reingold_layout(G)
        pos=nx.circular_layout(G)
        pos=nx.random_layout(G)
        pos=nx.spectral_layout(G)
        pos=nx.spring_layout(G)
        '''

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
        labels=labeldict,with_labels=True,node_size=500,
         linewidths=1, font_size=5, font_weight='bold',width=weights, edge_color=colors,node_color=gradientColor,
        )
        nx.draw_networkx_edge_labels(G, pos,edge_labels=edge_labels,label_pos=0.35, font_size=3  )
        print("The figures are saved: ", input.niche_pred_outdir+'Niche_interactions_with_edge_weights_R'+str(input.Radius)+'.'+saveas)
        fig.savefig(input.niche_pred_outdir+'Niche_interactions_with_edge_weights_R'+str(input.Radius)+'.'+saveas,bbox_inches='tight',transparent=transparent_mode,dpi=300)
        if showit:
            pass
        else:
            plt.close('all')
