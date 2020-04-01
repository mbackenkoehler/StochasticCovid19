import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import imageio
import glob
import random


########################################################
# Random graph models
########################################################

# nice grapkh for visuals
def geom_graph(node_num=350):
    radius = 0.07
    seed = 42
    for _ in range(1000):
        G = nx.random_geometric_graph(node_num, radius, seed=seed)
        node_num = G.number_of_nodes()
        pos = nx.get_node_attributes(G, 'pos')
        node_pos = [(pos[i][0], pos[i][1]) for i in range(node_num)]
        G = nx.convert_node_labels_to_integers(G)
        edges = [e for e in G.edges]
        if nx.is_connected(G):
            return G, node_pos
        radius *= 1.1
        seed += 1
    print('failed graph generation')


def fuzzy_geom_graph(size, radius, deg, ret_coords=True, force_connected=True):
    for _ in range(1000):
        # sample coordinates
        x, y = coords = np.random.rand(2, size) / radius

        # build the adjacency matrix
        adj = np.zeros((size, size)).astype(np.bool)
        for i, (xi, yi, di) in enumerate(zip(x, y, deg)):
            # sample neighbors based on euclidian distance
            p = np.exp(-np.sqrt((xi - x) ** 2 + (yi - y) ** 2))
            other_nodes = [k for k in range(size) if k != i]
            p = p[other_nodes]
            p /= p.sum()
            neighbors = np.random.choice(other_nodes, size=di, replace=False, p=p)
            adj[i, neighbors] = True
        adj |= adj.T

        G = nx.from_numpy_array(adj)
        if not force_connected or nx.is_connected(G):
            return G, coords.T
    print('failed graph generation')

# seed does not really work
def power_law_graph(num_nodes=100, gamma=2.2, min_truncate=2, max_truncate = None, seed = 42):
    degree_distribution = [0]+[(k+1)**(-gamma) for k in range(num_nodes)]
    degree_distribution[:min_truncate] = [0.0]*min_truncate
    if max_truncate is not None:
        # max truncate and everything larger is zero
        degree_distribution[max_truncate:] = [0.0] * (len(degree_distribution)-max_truncate)
    assert(len(degree_distribution) == num_nodes+1)
    z = np.sum(degree_distribution)
    degree_distribution = [p/z for p in degree_distribution]
    while True:
        seed += 1
        np.random.seed(seed)
        degee_sequence = [np.random.choice(range(num_nodes+1), p=degree_distribution) for _ in range(num_nodes)]
        if np.sum(degee_sequence) % 2 == 0:
            break

    np.random.seed(None)

    for seed in range(10000):
        seed += 42
        contact_network = nx.configuration_model(degee_sequence, create_using=nx.Graph)
        for n in contact_network.nodes():
            try:
                contact_network.remove_edge(n, n)  # hack, how to remove self-loops in nx 2.4??
            except:
                pass
        if nx.is_connected(contact_network):
            return contact_network, None
