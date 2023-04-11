import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
#from tensorflow.data import Dataset
tf.config.run_functions_eagerly(True)

class label_gen:
    def __init__():
        h5files = dict()
        h5files["ES"]=sorted(glob.glob("../h5data/ES/*h5"))
        h5files["CC"]=sorted(glob.glob("../h5data/CC/*h5"))
        ES=h5files["ES"]
        CC=h5files["CC"]
        print(ES[0])
        fes = h5py.File(ES[0],'r+')   
        print(fes['images'].keys())
        print("images/img",fes['images/img'])
        print(fes['images'].keys())

        #preprocessing

        
        fes = h5py.File(ES[1],'r+')   
        iid=4
        print(fes['images'].items())
        for k in self.fes['images'].keys():
            if 'img' in k: continue
            print(k,fes[f'images/{k}'][iid].shape,fes[f'images/{k}'][iid])
            #print( label_gen(fes['images'] ) )
    def label_gen(self, h5f ):
        matched = ( h5f['labelu'][()][::,0] == h5f['labelv'][()][::,0]) and (h5f['labelv'][()][::,0] == h5f['labelz'][()][::,0])
        n_nu, n_lepton, n_photon, n_proton, n_neutron, n_meson, nu_nucleus = np.array( list(zip(*h5f['particle_counter'][()])) )
        neutrino_P = h5f['neutrino_P'][()]
        neutrino_dir = normalize(neutrino_P[::,(0,1,2)],axis=1)

        part_P = h5f['part_P'][()]
        part_dir=normalize(part_P[::,(0,1,2)],axis=1)

        isES = n_nu == 2
        isRad = n_nu == 0
        isCC = np.array([ not (x or y) for x, y in zip(isES,isRad)])
        code = np.transpose( np.array([isES,isCC,isRad]) )    
        tpcside =  h5f['TPCID'][()][::,0]%2
        #return (matched,isES,tpcside)

        return code, matched, tpcside, neutrino_P,part_P    



class preprocessor:
    def __init__(self, image_size = (100,41,3), nclasses = 2, class_weight_array = None ):
        self.__image_size = image_size
        self.__nclasses = nclasses
        self.__class_weight = None
        if class_weight_array is None:
            class_weight_array = [1 for i in range(self.nclasses)]
        self.set_weight(class_weight_array)
       

    @property
    def image_size(self):
        return self.__image_size
    @image_size.setter
    def image_size(self, image_size):
        self.__image_size = image_size
    
    @property
    def nclasses(self):
        return self.__nclasses
    @nclasses.setter
    def nclasses(self, nclasses):
        self.__nclasses = nclasses
    
    @property
    def class_weight(self):
        return self.__class_weight
    @class_weight.setter
    def class_weight(self, class_weight):
        self.__class_weight = class_weight
    
    def set_weight(self, weight_array):
        self.__class_weight = {i:w for i,w in enumerate(weight_array)}

    def get_axis_size_list(self,X):
        old_dim = list(X.shape)
        if len(old_dim) != len(self.image_size):
            raise Exception("old_dim and image_size must have equal dimensions")
        
        size_list = []
        for i in range(len(old_dim)):
            if old_dim[i] != self.image_size[i]:
                size_list.append((i, self.image_size[i]))
        return size_list

            
            
            
    def resize_sum(self, X,new_size=41, axis=0):
        new_dim = list(X.shape)
        new_dim[axis]=new_size

        if  X.shape[axis] <= new_dim[axis]:
            return data
        combine_n =  X.shape[axis]//new_dim[axis]
        new_X = np.zeros(new_dim)
        for i in range(new_dim[axis]):
            index_min = i*combine_n
            index_max = index_min+combine_n+1
            values = np.sum(X[index_min:index_max],axis=axis)
            new_X[i]= values

        return new_X.astype(np.float32)


    def expand(self, y):
        if self.nclasses ==2 :
            #y = np.append(y, [(y[0]==y[1]==y[2]==False)])
            isES= (y[0]==1)
            y = np.array([isES, not isES])  
        else:
            y = np.array(y)
        return y

    def scaleNonZero(self, X):
        vcut=1
        X_dense = X
        X_dense[abs(X_dense)<vcut]=0
        for i in range(3):
            X1 = X[::,::,i][X[::,::,i]!=0]
            vMin, vMax, vMean = X1.min(), X1.max(), X1.mean()
            stdev = np.std(X1)
            N = len(X1)
            adj_stdev = max(stdev, 1/np.sqrt(N))
            #print(vMin,vMax,vMean)
            #scaling: 
            X_dense[::,::,i] = X[::,::,i]-vMean
            X_dense[::,::,i][X_dense[::,::,i]== -vMean] = 0
            X_dense[::,::,i][X_dense[::,::,i]!=0]/=adj_stdev
        X_dense[abs(X_dense)<vcut]=0
        
        res = X_dense
        for i, size_i in self.get_axis_size_list(X_dense):
            res = self.resize_sum(res,size_i,i)
        return res

    def inverse(self, x):
        x[::,::,2] = x[::-1,::,2]
        return x
    
    def get_weight(self,y):
        w =  np.float32(sum([v*self.class_weight[k] for k,v in enumerate(y)]))
        # print(y,w,self.class_weight)

        return w

    def Normalize(self,  X ,y,tpc,nuP,eP ):
        Xd =tf.sparse.to_dense( tf.cast(X, tf.float32),default_value=0.)
        X_dense = tf.numpy_function(self.scaleNonZero,[Xd],tf.float32)
        y = tf.numpy_function(self.expand,[y],tf.bool)
        return X_dense,y
                                                

    def NormalizeInv(self,  X ,y,tpc,nuP,eP ):
        X_dense, y = self.Normalize( X ,y,tpc,nuP,eP )
        X_dense = tf.numpy_function(self.inverse,[X_dense],tf.float32)
        return X_dense,y
    
    def NormalizeWWeight(self,  X ,y,tpc,nuP,eP ):
        Xd =tf.sparse.to_dense( tf.cast(X, tf.float32),default_value=0.)
        X_dense = tf.numpy_function(self.scaleNonZero,[Xd],tf.float32)
        y1 = tf.numpy_function(self.expand,[y],tf.bool)
        w = tf.numpy_function(self.get_weight,[y],tf.float32)
        return X_dense,y1, w

    def NormalizeInvWWeight(self,  X ,y,tpc,nuP,eP ):
        X_dense, y, w = self.NormalizeWWeight( X ,y,tpc,nuP,eP )
        X_dense = tf.numpy_function(self.inverse,[X_dense],tf.float32)
        return X_dense,y, w