import struct
import numpy as np
from sklearn.metrics import confusion_matrix
from numpy.random import *
from scipy.stats import invwishart
import matplotlib.pyplot as plt
import os

ExperimentName = "toy"

#Binary read-write
def readMat(filename):
    with open(filename,"rb") as fp:
        r = struct.unpack('I', bytearray(fp.read(4)))[0]
        m = struct.unpack('I', bytearray(fp.read(4)))[0]
        print r
        print m
        x = np.zeros((r,m))
        for i in range(0,r):
            for j in range(0,m):
                x[i,j] =  struct.unpack('d',bytearray(fp.read(8)))[0]
    return x
    
def writeMat(filename,x):
    with open(filename,"wb") as fp:
        (r,m) = np.array(x.shape).astype('int32')
        fp.write( struct.pack('i', r))
        fp.write( struct.pack('i', m))
        for i in range(0,r):
            for j in range(0,m):
                fp.write(struct.pack('d', x[i,j]))


#%% Create Data    
D=2
NCOMP = 5
NPOINTS = 1000
S = 10
mus = multivariate_normal(np.zeros(D),np.eye(D),NCOMP)
sigmas = [invwishart.rvs((D+2),np.eye(D)/S) for i in range(0,NCOMP)]
labels = randint(0,NCOMP,NPOINTS)
x = np.zeros((NPOINTS,D));
for i in range(0,NPOINTS):
    x[i,:] = multivariate_normal(mus[labels[i],:],sigmas[labels[i]]);

plt.scatter(x[:,0],x[:,1],c=labels)
plt.title('Original Labels')
plt.show()
filename = "Experiments\\"+ExperimentName+"\\toy.matrix"
writeMat(filename,x);


#%%Run executable 
os.system("dpsl.exe " + filename);
likelihood = readMat(filename + ".likelihood")
plt.plot(likelihood)
plt.show()
predsamples = readMat(filename + ".labels")
pred = predsamples[9,:].astype('int'); # See Matlab code for combining multiple label samples
plt.scatter(x[:,0],x[:,1],c=pred)
plt.title('Sampled Labels')
plt.show()
print("Confussion Matrix : \n")
print(confusion_matrix(labels,pred))