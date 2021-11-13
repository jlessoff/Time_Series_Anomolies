
import igraph as ig
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import operator
from functools import reduce
from sklearn import cluster
from sklearn import metrics
import random
# import networkx as nx
# You should also check

# Creating a graph from edge list
polbooks_net = ig.Graph.Read_Edgelist(f = "Data/polbooks-edges.txt", \
                                       directed = False)
print(polbooks_net)

# Creating a graph from adjacency matrix
fb_net = ig.Graph.Read_Adjacency(f="Data/Amherst41.adjacency",\
                                     sep="\t")
print(fb_net)

# Importing a graph object
lesmis_net = ig.Graph.Read_GML(f="Data/lesmiserables.gml")
print(lesmis_net)

# Attributes of a graph
lesmis_net.ecount()
lesmis_net.vcount()

lesmis_net.vs.attributes()
lesmis_net.vs['id']
lesmis_net.vs['label']
lesmis_net.es.attributes()
lesmis_net.es['value']

## Add attributes to edges and vertices
polbooks_info = pd.read_csv('Data/polbooks-nodes.txt', sep=" ")
polbooks_net.vs['label'] = polbooks_info['label']
polbooks_net.vs['value'] = polbooks_info['value']
print(polbooks_net)

## Visualizing a graph
p = ig.plot(lesmis_net, layout="random", bbox=(1000,1000), margin=30)
p.save('plot1.png')
p = ig.plot(lesmis_net, layout="circle", bbox=(1000,1000), margin=30)
p.save('plot1.png')
p = ig.plot(lesmis_net, layout="fr", bbox=(1000,1000), margin=30)
p.save('plot1.png')

# Visualizing a graph with some customization
visual_style = {}
visual_style["vertex_size"] = 1
visual_style["vertex_color"] = "blue"
visual_style["vertex_label"] = lesmis_net.vs["label"]
visual_style["vertex_label_size"] = 12
visual_style["vertex_label_color"] = "blue"
visual_style["edge_width"] = np.array(lesmis_net.es['value'])*5/max(np.array(lesmis_net.es['value']))
visual_style["layout"] = "fr"
visual_style["bbox"] = (1000, 1000)
visual_style["margin"] = 30
p = ig.plot(lesmis_net, **visual_style)
p.save('plot1.png')

random.seed(2021)
p = ig.plot(lesmis_net, **visual_style)
p.save('plot1.png')

# Check if a graph is connected
lesmis_net.is_connected()
polbooks_net.is_connected()
fb_net.is_connected()

# Statistical summaries and global characteristics
lesmis_net.density()

lesmis_net.transitivity_undirected()

lesmis_net.diameter()

lesmis_net.radius()

lesmis_net.girth()

lesmis_net.cohesion()

# Compute the shortest path between two nodes
lesmis_net.vs["name"] = lesmis_net.vs['label']
lesmis_net.get_shortest_paths(v = "MotherPlutarch",
               to = "Myriel", mode="out", output='vpath')

# All shortest paths
lesmis_sp = lesmis_net.shortest_paths()
sb.histplot(reduce(operator.concat, lesmis_sp),\
            stat="density", bins=100)
plt.show()

# Hierachical clustering on shortest paths distances
silhouette_scores = []
CS_scores = []
for k in range(2,21):
    hac = cluster.AgglomerativeClustering(n_clusters=k, linkage="average",\
                                affinity="precomputed")
    cluster_found = hac.fit_predict(np.matrix(lesmis_sp))
    silhouette_scores.append(metrics.silhouette_score(np.matrix(lesmis_sp), hac.labels_))
    CS_scores.append(metrics.calinski_harabasz_score(np.matrix(lesmis_sp), hac.labels_))
plt.plot(range(2,21), silhouette_scores)
plt.close()
plt.plot(range(2,21), CS_scores)
plt.close()

hac = cluster.AgglomerativeClustering(n_clusters=8, linkage="average", \
                        affinity="precomputed")
cluster_found = hac.fit_predict(np.matrix(lesmis_sp))
pal = ig.drawing.colors.ClusterColoringPalette(8)
visual_style["vertex_color"] = pal.get_many(cluster_found)
visual_style["vertex_label_color"] = pal.get_many(cluster_found)
random.seed(2021)
p = ig.plot(lesmis_net, **visual_style)
p.save('plot1.png')


# Local characteristics
lesmis_net.degree()
sb.distplot(lesmis_net.degree(), bins=10, kde=True)
plt.show()

lesmis_net.betweenness()
sb.distplot(lesmis_net.betweenness(), bins=10, kde=True)
plt.show()

lesmis_net.eccentricity()
lesmis_net.closeness()
lesmis_net.strength()

# Colour nodes according to vertex characteristics
pal = ig.GradientPalette("white", "red", max(lesmis_net.degree())+1)
visual_style["vertex_color"] = pal.get_many(lesmis_net.degree())
visual_style["vertex_label_color"] = pal.get_many(lesmis_net.degree())
p = ig.plot(lesmis_net, **visual_style)
p.save('plot1.png')




