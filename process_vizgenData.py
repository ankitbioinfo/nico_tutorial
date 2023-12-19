
import pandas as pd
import numpy as np

mydir='./vizgenData/'

filename=mydir+'cell_by_gene_vizgen.csv'
df=pd.read_csv(filename,sep=',')
expression=df.to_numpy()
print('shape',expression.shape)


cellname=[]
d={}
temp=expression[:,0]
fw=open(mydir+'cellname.txt','w')
for i in range(len(temp)):
    name='cell'+str(i)
    cellname.append(name)
    d[str(temp[i])]=name
    #d[str(temp[i])]=temp[i]
    fw.write(str(temp[i])+'\t'+name+'\n')


LRgenes=[]
gene_index=[]
for i in range(1,len(df.columns)):
    if df.columns[i].find('Blank')==-1:
        gene_index.append(i)
        LRgenes.append(df.columns[i])
    #print(i,df.columns[i])

print(len(gene_index))


mat=np.transpose(expression[:,gene_index])

ndf=pd.DataFrame(mat,index=LRgenes)

ndf.to_csv(mydir+'gene_by_cell.csv',index=True, index_label="GENEID",  header=cellname)

f2=open(mydir+'cell_metadata.csv')
f2.readline()
data=[]
for line in f2:
    l=line.split(',')
    data.append(l)

f3=open(mydir+'tissue_positions_list.csv','w')
f3.write('barcode,Xcoord,Ycoord\n')
temp=[]
#EntityID,fov,volume,center_x,center_y,min_x,min_y,max_x,max_y,anisotropy,
for i in range(len(data)):
    f3.write(d[data[i][0]]+','+ str(data[i][3])+','+str(data[i][4]) +'\n')
    temp.append(d[data[i][0]])

print('done')
temp=np.array(temp)
cellname=np.array(cellname)

print('equal',np.array_equal(temp,cellname))
