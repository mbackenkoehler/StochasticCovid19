import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def viz_simulate(G, time_points, storage, model, node_pos=None, outpath='output_simulation_NUM.jpg', storage_rec=None,
                 show_title=False):

    os.system('mkdir vis_out')
    if node_pos is None:
        # pos=nx.spring_layout(G,seed=11)
        pos = nx.spectral_layout(G)
        node_pos = [(pos[i][0], pos[i][1]) for i in range(G.number_of_nodes())]

    #storage = storage/np.max(storage)
    for i in range(storage.shape[1]):
        node_labels = list(storage[:,i])
        plt.clf()

        # nodes
        for node in G.nodes():
            c = storage[node, i]
            c_rgb = model.get_colors()[int(c+0.000001)]
            s = 1.0 / len(node_pos) * 150 * 30
            plt.scatter([node_pos[node][0]], [node_pos[node][1]], s=s, alpha=0.8, zorder=15, c=c_rgb)

        # edges
        lw = 1.0 / len(G.edges()) * 600
        for e in G.edges:
            pos_v1 = node_pos[e[0]]
            pos_v2 = node_pos[e[1]]
            plt.plot([pos_v1[0], pos_v2[0]], [pos_v1[1], pos_v2[1]], c='black', alpha=0.5, zorder=10, lw=lw)

        title = str(time_points[i]) + '     '
        if show_title:
            plt.title(title[0:5])
        else:
            (plt.gca()).spines['top'].set_visible(False)
            (plt.gca()).spines['right'].set_visible(False)
            (plt.gca()).spines['bottom'].set_visible(False)
            (plt.gca()).spines['left'].set_visible(False)
        (plt.gca()).set_yticklabels([])
        (plt.gca()).set_xticklabels([])
        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig('vis_out/' + outpath.replace('NUM', str(10000 + i)), bbox_inches='tight', dpi=300)