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



def geometric_configuration_model(degree_sequence, coordinates=None):
    import itertools
    G = nx.Graph()
    num_nodes = len(degree_sequence)
    if coordinates is None:
        coordinates = np.random.rand(2, num_nodes)

    stubs = [[v_i]*degree for v_i, degree in enumerate(degree_sequence)]
    stubs = list(itertools.chain.from_iterable(stubs))
    if len(stubs) % 2 != 0:
        raise ValueError("Sum of degree sequence must be even.")

    random.shuffle(stubs)
    for v_i in range(num_nodes):
        G.add_node(v_i)
        G.nodes[v_i]['pos'] = coordinates[:,v_i]


    for _ in range(len(stubs)*2):
        if len(stubs) == 0:
            break
        v1, v2 = stubs[:2]
        if v1 == v2:
            random.shuffle(stubs)
        elif G.has_edge(v1,v2):
            random.shuffle(stubs)
        else:
            G.add_edge(v1,v2)
            stubs = stubs[2:]


    max_steps = len(G.edges)*10000
    for i_step in range(max_steps):
        edges = list(G.edges())
        e1 = random.choice(edges)
        e2 = random.choice(edges)
        if len(set(list(e1) + list(e2))) == 4:  # make sure they do not share nodes
            e1_list = list(e1)
            e2_list = list(e2)
            random.shuffle(e1_list)  # more suble trick to avoid bias (dont rewire primarily lower nodes and higher nodes with each other)
            random.shuffle(e2_list)
            new_edge1 = (e1_list[0], e2_list[0])
            new_edge2 = (e1_list[1], e2_list[1])

            if G.has_edge(*new_edge1) or G.has_edge(*new_edge2):
                continue

            v1 = e1_list[0]
            v2 = e1_list[1]
            v3 = e2_list[0]
            v4 = e2_list[1]

            v1_pos = coordinates[:,v1]
            v2_pos = coordinates[:,v2]
            v3_pos = coordinates[:,v3]
            v4_pos = coordinates[:,v4]

            cost = np.linalg.norm(v1_pos - v2_pos) + np.linalg.norm(v3_pos - v4_pos)
            cost_new = np.linalg.norm(v1_pos - v3_pos) + np.linalg.norm(v2_pos - v4_pos)

            cost= cost**5
            cost_new = cost_new**5

            ratio = cost_new/(cost+cost_new)

            #if ratio > random.random():
            if cost_new > cost:
                continue

            G.add_edge(*new_edge1)
            G.add_edge(*new_edge2)
            G.remove_edge(*e1)
            G.remove_edge(*e2)

            if len(stubs) > 0:
                v1, v2 = stubs[:2]
                if v1 != v2 and not G.has_edge(v1, v2):
                    G.add_edge(v1, v2)
                    stubs = stubs[2:]

            if (i_step > max_steps/10) and nx.is_connected(G) and len(stubs)==0:  # note that we need burn in period
                return G, [(coordinates[0][i], coordinates[1][i]) for i in range(G.number_of_nodes())]
            if (i_step > max_steps/2) and nx.is_connected(G):
                return G, [(coordinates[0][i], coordinates[1][i]) for i in range(G.number_of_nodes())]
    print('Generation failed')



def test_geometric_configuration_model():
    plt.clf()
    G = geometric_configuration_model([3]*500)
    pos=nx.spring_layout(G)
    pos = {v_i: G.nodes[v_i]['pos'] for v_i in G.nodes()}
    nx.draw_networkx_edges(G,pos=pos, width=0.5)
    nx.draw_networkx_nodes(G,pos=pos, alpha=0.5, node_size=40)
    plt.savefig('example_geom_config.pdf')
