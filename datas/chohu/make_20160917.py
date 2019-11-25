import csv
import numpy as np
import numpy.matlib
import tqdm

x1=np.array([ v for v in csv.reader(open("./chohu_01.csv", "r")) if len(v)!= 0])
x2=np.array([ v for v in csv.reader(open("./chohu_02.csv", "r")) if len(v)!= 0])
#x=np.array([mol for mol in [[int(nucl) for nucl in atom if len(nucl) != 0 and nucl.strip().isdigit()] for atom in csv.reader(open("./01_納品物/01_調布市_総数.csv", "r"))] if len(mol) != 0])
#print(x.shape[0])
x1=np.delete(x1,0,0)
x2=np.delete(x2,0,0)
x=np.append(x1,x2,axis=0)
#print(x.shape[0])
id=np.array([ v for v in csv.reader(open("id to num_small.csv", "r")) if len(v)!= 0])
id=np.delete(id,0,0)
print(x.shape[0])
for i in range(3):
    x=np.delete(x,4,1)
for i in range(x.shape[0]):
    if(x[i,3]=='ChofuSta1'):
        x[i,3]='533934832'
    elif(x[i,3]=='ChofuSta2'):
        x[i,3]='666666666'
x=x.astype(np.int32)
id=id.astype(np.int32)

f = open('all.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
for i in tqdm.trange(x.shape[0]):
    if x[i,3] in id[:,1]:
         writer.writerow(x[i,:])
f.close()

x=np.array([ v for v in csv.reader(open("all.csv", "r")) if len(v)!= 0])
print(x.shape[0])
x=x.astype(np.int32)
#for i in tqdm.trange(x.shape[0]):
    #print(i)
    #print(x[*i:117*i+117,4])

f = open('allarea.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
for i in tqdm.trange(x.shape[0]):
    writer.writerow(x[29*i:29*i+29,4])
f.close()

"""
out=np.zeros([id.shape[0]+1,9])
out[:,0]=np.arange(id.shape[0]+1)

for k in range(id.shape[0]):
    sum=np.zeros(8)
    n=np.zeros(8)
    for i in range(x.shape[0]):
        if(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]<=14 and x[i,2]==1700 and x[i,3]==id[k,1]):
            n[0]+=1
            sum[0]+=x[i,4]
        elif(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]==15 and x[i,2]==1700 and x[i,3]==id[k,1]):
            n[1]+=1
            sum[1]+=x[i,4]
        elif(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]==16 and x[i,2]==1700 and x[i,3]==id[k,1]):
            n[2]+=1
            sum[2]+=x[i,4]
        elif(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]==17 and x[i,2]==1700 and x[i,3]==id[k,1]):
            n[3]+=1
            sum[3]+=x[i,4]
        elif(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]<=14 and x[i,2]==1800 and x[i,3]==id[k,1]):
            n[4]+=1
            sum[4]+=x[i,4]
        elif(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]==15 and x[i,2]==1800 and x[i,3]==id[k,1]):
            n[5]+=1
            sum[5]+=x[i,4]
        elif(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]==16 and x[i,2]==1800 and x[i,3]==id[k,1]):
            n[6]+=1
            sum[6]+=x[i,4]
        elif(20170929<=x[i,0] and x[i,0]<=20180927 and x[i,1]==17 and x[i,2]==1800 and x[i,3]==id[k,1]):
            n[7]+=1
            sum[7]+=x[i,4]

    for i in tqdm(range(8)):
        if(n[i]>0):
            out[k+1,i+1]=sum[i]/n[i]
            out[k+1,i+1]=np.sqrt((x[i,4]-out[k+1,i+1])**2/n[i])
    print(out[k,:])

f = open('sweeka.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
for i in range(out.shape[0]):
     writer.writerow(out[i,:])
f.close()
"""