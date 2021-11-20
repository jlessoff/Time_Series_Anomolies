
import igraph as ig
import pandas as pd
import numpy as np
from sklearn import cluster
import random
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

# Creating a graph from edge list
polbooks_net = ig.Graph.Read_Edgelist(f = "Data/polbooks-edges.txt", \
                                       directed = False)
# print(polbooks_net)

# Creating a graph from adjacency matrix
fb_net = ig.Graph.Read_Adjacency(f="Data/Amherst41.adjacency",\
                                     sep="\t")
# print(fb_net)

# Importing a graph object
lesmis_net = ig.Graph.Read_GML(f="Data/lesmiserables.gml")
# print(lesmis_net)

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


random.seed(2021)


# Check if a graph is connected
lesmis_net.is_connected()
polbooks_net.is_connected()
fb_net.is_connected()

# Statistical summaries and global characteristics

# All shortest paths
lesmis_sp = lesmis_net.shortest_paths()

hac = cluster.AgglomerativeClustering(n_clusters=8, linkage="average", \
                        affinity="precomputed")
cluster_found = hac.fit_predict(np.matrix(lesmis_sp))
pal = ig.drawing.colors.ClusterColoringPalette(8)
visual_style["vertex_color"] = pal.get_many(cluster_found)
visual_style["vertex_label_color"] = pal.get_many(cluster_found)
random.seed(2021)
p = ig.plot(lesmis_net, **visual_style)
p.save('plot1.png')

lesmis_sp = lesmis_net.shortest_paths()

hac = cluster.AgglomerativeClustering(n_clusters=2, linkage="average", \
                        affinity="euclidean")
cluster_found = hac.fit_predict(np.matrix(lesmis_sp))
pal = ig.drawing.colors.ClusterColoringPalette(8)
visual_style["vertex_color"] = pal.get_many(cluster_found)
visual_style["vertex_label_color"] = pal.get_many(cluster_found)
random.seed(2021)
p = ig.plot(lesmis_net, **visual_style)
p.save('euclidean.png')

hac = cluster.AgglomerativeClustering(n_clusters=2, linkage="average", \
                        affinity="cosine")
cluster_found = hac.fit_predict(np.matrix(lesmis_sp))
pal = ig.drawing.colors.ClusterColoringPalette(8)
visual_style["vertex_color"] = pal.get_many(cluster_found)
visual_style["vertex_label_color"] = pal.get_many(cluster_found)
random.seed(2021)
p = ig.plot(lesmis_net, **visual_style)
p.save('cosine.png')

hac = cluster.AgglomerativeClustering(n_clusters=8, linkage="average", \
                        affinity="euclidean")
cluster_found = hac.fit_predict(np.matrix(lesmis_sp))
pal = ig.drawing.colors.ClusterColoringPalette(8)
visual_style["vertex_color"] = pal.get_many(cluster_found)
visual_style["vertex_label_color"] = pal.get_many(cluster_found)
random.seed(2021)
p = ig.plot(lesmis_net, **visual_style)
p.save('euclidean8.png')

hac = cluster.AgglomerativeClustering(n_clusters=8, linkage="average", \
                        affinity="cosine")
cluster_found = hac.fit_predict(np.matrix(lesmis_sp))
pal = ig.drawing.colors.ClusterColoringPalette(8)
visual_style["vertex_color"] = pal.get_many(cluster_found)
visual_style["vertex_label_color"] = pal.get_many(cluster_found)
random.seed(2021)
p = ig.plot(lesmis_net, **visual_style)
p.save('cosine8.png')

