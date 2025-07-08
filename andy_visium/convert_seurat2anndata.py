import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np


#df=pd.read_csv('umap_embeddings.csv')
#umap = df.to_numpy()

df=pd.read_csv('coordinates.csv')
coordinates=df.to_numpy()

df=pd.read_csv('cell_metadata.csv')
#df = df[['seurat_clusters']]
spot_class=df[['spot_class']]
first_type=df[['first_type']]
second_type=df[['second_type']]

spot_class=spot_class.to_numpy()
first_type=first_type.to_numpy()
second_type=second_type.to_numpy()

df=pd.read_csv('pca_embeddings.csv')
pca=df.to_numpy()

adata = sc.read('counts_data.csv').transpose()
#adata2 = sc.read('counts_data2.csv').transpose()

#adata = ad.concat([adata1, adata2], join="inner")#.X.toarray()

#print(umap.shape,meta.shape,pca.shape)
#print(adata1.obs_names)
#print(adata2.obs_names)
#print("equal 1",np.array_equal(adata.obs_names,umap[:,0]))
#print("equal 2",np.array_equal(adata.obs_names,meta[:,0]))


#adata.obsm['X_umap']=umap[:,1:3].astype(float)

#print(meta)

typeclass=np.unique(spot_class)

for i in range(len(typeclass)):
    ind=np.where(spot_class==typeclass[i])[0]
    print("class", typeclass[i],len(ind))


annot=[]
myfirst=[]
mysecond=[]
index=[]
for i in range(len(spot_class)):
    flag=0
    if spot_class[i]=='doublet_uncertain':
        #print("un",first_type[i],second_type[i])
        if second_type[i,0]=='Not_annotated':
            annot.append(first_type[i,0])
        else:
            annot.append(second_type[i,0])
        flag=1
    if spot_class[i]=='singlet':
        #print("sing",first_type[i],second_type[i])
        annot.append(second_type[i,0])
        flag=1
    if spot_class[i]=='doublet_certain':
        #print("sing",first_type[i],second_type[i])
        if first_type[i,0]=='Not_annotated':
            annot.append(second_type[i,0])
        else:
            annot.append(first_type[i,0])
        flag=1

    if flag==1:
        index.append(i)
        if first_type[i,0]=='Not_annotated':
            myfirst.append('CM')
        else:
            myfirst.append(first_type[i,0])

        if second_type[i,0]=='Not_annotated':
            mysecond.append('Fibroblasts')
        else:
            mysecond.append(second_type[i,0])



annot=np.array(annot)
myfirst=np.array(myfirst)
mysecond=np.array(mysecond)
#annot=np.reshape(annot,len(annot),1)

adata.obsm['X_pca']=pca[:,1:].astype(float)
adata.obsm['spatial']=coordinates[:,1:].astype(float)

print(adata)

adataNew=adata[index].copy()
print(annot.shape)
adataNew.obs['RCTD_cluster']=annot     #meta[:,0].astype(str)
adataNew.obs['RCTD_first']=myfirst
adataNew.obs['RCTD_second']=mysecond
#adata.obs['state']=meta[:,5].astype(str)
print(adataNew)


def count_freq(annot):
    d={}
    for i in range(len(annot)):
        name=annot[i]
        if name in d:
            d[name]+=1
        else:
            d[name]=1

    for key in d:
        print(key,d[key])



print("\nannot")
count_freq(annot)
print("\n\nmyfirst")
count_freq(myfirst)
print("\n\nmysecond")
count_freq(mysecond)

adataNew.raw=adataNew.copy()

sc.pp.neighbors(adataNew)
sc.tl.umap(adataNew)
sc.pl.umap(adataNew)

adataNew.X=adataNew.raw.X

print(adataNew.X)
print(adataNew.var_names)
print(adataNew.obs_names)

adataNew.write_h5ad('S_spatial_day7_r1.h5ad')


#myannot='seurat_cluster'
#sc.pl.umap(adata, color=[myannot], title=[""],wspace=0.4,#color_map='viridis',
#           show=False, save='_side.png')

#sc.pl.umap(adata, color=[myannot], title=[""],wspace=0.4,#color_map='viridis',
#           legend_loc='on data',
#           legend_fontsize=5,
#           show=False, save='_ondata.png')
