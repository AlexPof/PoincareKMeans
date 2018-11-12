import numpy as np
from poincarekmeans import PoincareKMeans

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

points = np.loadtxt("./poincare_coordinates.csv", delimiter=';')
colormap = get_cmap('jet')

model = PoincareKMeans(n_clusters=10,n_init=100,max_iter=200,tol=1e-10,verbose=True)
model.fit(points)
clust_labels = model.labels_
centroids = model.cluster_centers_

colors = [colormap(x/model.n_clusters) for x in clust_labels]

plt.figure(figsize=(8,8))
plt.scatter(points[:,0],points[:,1],alpha=0.3,c=colors)
for i in range(model.n_clusters):
    plt.scatter(centroids[i,0],centroids[i,1],alpha=1.0,c="red")
plt.show()