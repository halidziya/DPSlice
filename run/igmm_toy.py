import struct
import numpy as np
from sklearn.metrics import f1_score

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
    
m = readMat("experiments\\toy\\mnist.matrix")
l = readMat("experiments\\toy\\mnist.matrix.likelihood")


t=np.array(range(0,60000))
t=t/6000
lbs = readMat("experiments\\toy\\mnist.matrix.labels")
lbs[-1,:]
f1_score(t,lbs[-1,:],average="macro")
cl = lbs[-1,:].transpose().astype('int')
confmat = sklearn.metrics.confusion_matrix(t,cl)
amap = np.max(confmat,1) # I need to solve munkres


#x = np.zeros((len(lines),100));
#for i in range(0,len(lines)):
#    x[i,:] = np.array(lines[i].split()[1:])
    