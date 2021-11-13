
import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Importing "Les misérables" graph
lesmis_net = ig.Graph.Read_GML(f="Data/lesmiserables.gml")
print(lesmis_net)

# Generate random graphs with the Erdos model
random.seed(2021)
B = 100
global_char = []
for i in range(B):
    random_graph = ig.Graph.Erdos_Renyi(n=lesmis_net.vcount(), \
                                        m=lesmis_net.ecount(), \
                                        directed=False, loops=False)
    if random_graph.is_connected():
        global_char.append([random_graph.density(),
                            random_graph.transitivity_undirected(),
                            random_graph.diameter()])

# How many of the random graphs are connected?
print(len(global_char))

# How to the density, transitivity and diameter compare with the real graph?
global_char = pd.DataFrame.from_records(data=global_char, \
                                        columns=['Density', 'Transitivity', 'Diameter'])

plt.hist(global_char['Density'])
plt.axvline(x=lesmis_net.density(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Transitivity'])
plt.axvline(x=lesmis_net.transitivity_undirected(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Diameter'])
plt.axvline(x=lesmis_net.diameter(), color='r', linewidth=2)
plt.show()

# Generate random graphs with the Barabasi - Albert model
random.seed(2021)
B = 100
global_char = []
for i in range(B):
    random_graph = ig.Graph.Barabasi(n=lesmis_net.vcount(), m=4)
    if random_graph.is_connected():
        global_char.append([random_graph.density(),
                            random_graph.transitivity_undirected(),
                            random_graph.diameter()])

# How many of the random graphs are connected?
print(len(global_char))

# How to the density, transitivity and diameter compare with the real graph?
global_char = pd.DataFrame.from_records(data=global_char, \
                                        columns=['Density', 'Transitivity', 'Diameter'])

plt.hist(global_char['Density'])
plt.axvline(x=lesmis_net.density(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Transitivity'])
plt.axvline(x=lesmis_net.transitivity_undirected(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Diameter'])
plt.axvline(x=lesmis_net.diameter(), color='r', linewidth=2)
plt.show()

# Generate random graphs using random permutations
random.seed(2021)
B = 100
iter = 100
global_char = []
for i in range(B):
    random_graph = lesmis_net.copy()
    random_graph.rewire(n=100 * lesmis_net.vcount())
    if random_graph.is_connected():
        global_char.append([random_graph.density(),
                            random_graph.transitivity_undirected(),
                            random_graph.diameter()])

# How many of the random graphs are connected?
print(len(global_char))

# How to the density, transitivity and diameter compare with the real graph?
global_char = pd.DataFrame.from_records(data=global_char, \
                                        columns=['Density', 'Transitivity', 'Diameter'])

plt.hist(global_char['Density'])
plt.axvline(x=lesmis_net.density(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Transitivity'])
plt.axvline(x=lesmis_net.transitivity_undirected(), color='r', linewidth=2)
plt.show()

plt.hist(global_char['Diameter'])
plt.axvline(x=lesmis_net.diameter(), color='r', linewidth=2)
plt.show()

# Compute a list of B betweenness distributions
random.seed(2021)
B = 100
iter = 100
bw_list = []
for i in range(B):
    random_graph = lesmis_net.copy()
    random_graph.rewire(n=100 * lesmis_net.vcount())
    if random_graph.is_connected():
        bw_list.append(random_graph.betweenness())

# How many of the random graphs are connected?
print(len(bw_list))

# How is the betweenness of the vertices related to a random graph?
# Empirical p-values
bw_list = np.array(bw_list)
p_high = []
p_low = []
for i in range(lesmis_net.vcount()):
    p_high.append(sum(lesmis_net.betweenness()[i] > bw_list[:, i]) / len(bw_list))
    p_low.append(sum(lesmis_net.betweenness()[i] < bw_list[:, i]) / len(bw_list))

color_vect = []
for i in range(lesmis_net.vcount()):
    if p_high[i] >= 0.95:
        color_vect.append("red")
    elif p_low[i] >= 0.95:
        color_vect.append("blue")
    else:
        color_vect.append("grey")

visual_style = {}
visual_style["vertex_size"] = 1
visual_style["vertex_color"] = color_vect
visual_style["vertex_label"] = lesmis_net.vs["label"]
visual_style["vertex_label_size"] = 10
visual_style["vertex_label_color"] = color_vect
visual_style["edge_width"] = np.array(lesmis_net.es['value']) * 5 / max(np.array(lesmis_net.es['value']))
visual_style["layout"] = "fr"
visual_style["bbox"] = (1000, 1000)
visual_style["margin"] = 20
p = ig.plot(lesmis_net, **visual_style)
p.save('plot1.png')
