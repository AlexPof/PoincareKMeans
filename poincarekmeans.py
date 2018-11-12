from random import sample
import numpy as np

## K-Means in the Poincare Disk model

class PoincareKMeans(object):
    def __init__(self,n_clusters=8,n_init=20,max_iter=300,tol=1e-8,verbose=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose =  verbose
        self.labels_=None
        self.cluster_centers_ = None
           

    def fit(self,X):
        n_samples = X.shape[0]
        self.inertia = None

        for run_it in range(self.n_init):
            centroids = X[sample(range(n_samples),self.n_clusters),:]
            for it in range(self.max_iter):
                distances = self._get_distances_to_clusters(X,centroids)
                labels = np.argmin(distances,axis=1)

                new_centroids = np.zeros((self.n_clusters,2))
                for i in range(self.n_clusters):
                    indices = np.where(labels==i)[0]
                    if len(indices)>0:
                        new_centroids[i,:] = self._hyperbolic_centroid(X[indices,:])
                    else:
                        new_centroids[i,:] = X[sample(range(n_samples),1),:]
                m = np.ravel(centroids-new_centroids, order='K')
                diff = np.dot(m,m)
                centroids = new_centroids.copy()
                if(diff<self.tol):
                    break
                
            distances = self._get_distances_to_clusters(X,centroids)
            labels = np.argmin(distances,axis=1)
            inertia = np.sum([np.sum(distances[np.where(labels==i)[0],i]**2) for i in range(self.n_clusters)])
            if (self.inertia == None) or (inertia<self.inertia):
                self.inertia = inertia
                self.labels_ = labels.copy()
                self.cluster_centers_ = centroids.copy()
                
            if self.verbose:
                print("Iteration: {} - Best Inertia: {}".format(run_it,self.inertia))
                          
    def fit_predict(self,X):
        self.fit(X)
        return self.labels_

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
    def predict(self,X):
        distances = self.transform(X)
        return np.argmin(distances,axis=1)
        
    def transform(self,X):
        return _get_distances_to_clusters(X,self.cluster_centers_)
    
    def _get_distances_to_clusters(self,X,clusters):
        n_samples,n_clusters = X.shape[0],clusters.shape[0]
        
        distances = np.zeros((n_samples,n_clusters))
        for i in range(n_clusters):
            centroid = np.tile(clusters[i,:],(n_samples,1))
            den1 = 1-np.linalg.norm(X,axis=1)**2
            den2 = 1-np.linalg.norm(centroid,axis=1)**2
            the_num = np.linalg.norm(X-centroid,axis=1)**2
            distances[:,i] = np.arccosh(1+2*the_num/(den1*den2))
        
        return distances
      
    def _poinc_to_minsk(self,points):
        minsk_points = np.zeros((points.shape[0],3))
        minsk_points[:,0] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[0]/(1-v[0]**2-v[1]**2))
        minsk_points[:,1] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[1]/(1-v[0]**2-v[1]**2))
        minsk_points[:,2] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: (1+v[0]**2+v[1]**2)/(1-v[0]**2-v[1]**2))
        return minsk_points

    def _minsk_to_poinc(self,points):
        poinc_points = np.zeros((points.shape[0],2))
        poinc_points[:,0] = points[:,0]/(1+points[:,2])
        poinc_points[:,1] = points[:,1]/(1+points[:,2])
        return poinc_points

    def _hyperbolic_centroid(self,points):
        minsk_points = self._poinc_to_minsk(points)
        minsk_centroid = np.mean(minsk_points,axis=0)
        normalizer = np.sqrt(np.abs(minsk_centroid[0]**2+minsk_centroid[1]**2-minsk_centroid[2]**2))
        minsk_centroid = minsk_centroid/normalizer
        return self._minsk_to_poinc(minsk_centroid.reshape((1,3)))[0]