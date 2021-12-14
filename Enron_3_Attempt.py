from Enron_Project_Data import graph_ds, positions, profiles_agg
import pandas as pd
import random
import igraph as ig
import matplotlib.pyplot as plt

import numpy as np
print(len(graph_ds))
sendrec=(graph_ds[['sender_id','destin_id']]).dropna()
sendrec= sendrec.astype({'sender_id': 'int64', 'destin_id':'int64'})
G2 = ig.Graph.DataFrame(directed=True,edges=sendrec)

# Attributes of a graph
G2.es['weight'] = graph_ds['count']

print(G2.vs.attributes())
# G2.vs['label']
print(G2.es.attributes())
#

###GRAPH STYLE
visual_style = {}
visual_style["vertex_size"] = 30
# np.array(G2.vs['name'])*30/max(np.array(G2.vs['name']))
visual_style["edge_arrow_size"]=0.3
visual_style["vertex_color"] = "pink"
visual_style["vertex_label"] = G2.vs['name']
visual_style["vertex_label_size"] = 20
visual_style["vertex_label_color"] = "black"
visual_style["edge_width"] = np.array(G2.es['weight'])*30/max(np.array(G2.es['weight']))
visual_style["layout"] = "fr"
visual_style["bbox"] = (2000, 2000)
visual_style["margin"] = 30
random.seed(2021)
p = ig.plot(G2, **visual_style)
p.save('directed_weighted_enron.png')


##CLUSTER STYLE
visual_style_clust = {}
visual_style_clust["edge_width"] = np.array(G2.es['weight'])*30/max(np.array(G2.es['weight']))
visual_style_clust["edge_color"] = "black"

##CLUSTER
community=G2.community_spinglass(weights=G2.es['weight'])
#https://igraph.org/python/doc/api/igraph.Graph.html#modularity
##SHOW CLUSTER MEMBERSHIP
print(community)
membership = community.membership
vc = ig.VertexClustering(G2, membership)
result = []
for c in vc:
    result.append(set([G2.vs[i] for i in c]))


CLUSTER= ig.plot(vc, bbox=(2400, 1400), vertex_label=G2.vs['name'], vertex_size=30, **visual_style_clust)
CLUSTER.save('cluster_enron.png')

print(community.modularity)


###RANDOM GRAPHS
random.seed(2021)
B = 100
global_char = []
for i in range(B):
    random_graph = ig.Graph.Erdos_Renyi(n=G2.vcount(), \
                                        m=G2.ecount(), \
                                        directed=False, loops=False)
    if random_graph.is_connected():
        global_char.append([random_graph.density(),
                            random_graph.transitivity_undirected(),
                            random_graph.diameter()])

# How many of the random graphs are connected?

global_char = pd.DataFrame.from_records(data=global_char, \
                                        columns=['Density', 'Transitivity', 'Diameter'])

plt.hist(global_char['Density'])
plt.axvline(x=G2.density(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Transitivity'])
plt.axvline(x=G2.transitivity_undirected(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Diameter'])
plt.axvline(x=G2.diameter(), color='r', linewidth=2)
plt.show()

# How many of the random graphs are connected?
print(len(global_char))


##B


