import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time, random, os
import seaborn as sns
import scipy
import glob
import pickle
import sys
import collections
from scipy.linalg import expm
import heapq
import copy


# model has to have states(), get_init_labeling(), and next_event(self, G, src_node, global_clock) method
class SISmodel:
    def __init__(self, infection_rate):
        self.infection_rate = infection_rate

    def states(self):
        return ['I', 'S']

    def get_init_labeling(self):
        init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def next_event(self, G, src_node, global_clock):
        event_id = G.nodes[src_node]['event_id']
        event_id += 1

        if G.nodes[src_node]['state'] == 'I':
            new_state = 'S'
            fire_time = -np.log(random.random())  # recov-rate is alsways 1
        else:
            new_state = 'I'
            inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] == 'I'])
            if inf_neighbors == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = inf_neighbors * self.infection_rate
                fire_time = -np.log(random.random()) / node_rate

        G.nodes[src_node]['event_id'] = event_id
        new_time = global_clock + fire_time
        return new_time, src_node, new_state, event_id



class SIRmodel:
    def __init__(self, infection_rate):
        self.infection_rate = infection_rate

    def states(self):
        return ['I', 'S', 'R']

    def get_init_labeling(self, G):
        init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def next_event(self, G, src_node, global_clock):
        event_id = G.nodes[src_node]['event_id']
        event_id += 1

        if G.nodes[src_node]['state'] == 'I':
            new_state = 'R'
            fire_time = -np.log(random.random())
        elif G.nodes[src_node]['state'] == 'S':
            new_state = 'I'
            inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] == 'I'])
            if inf_neighbors == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = inf_neighbors * self.infection_rate
                fire_time = -np.log(random.random()) / node_rate
        else:
            new_state = 'R'
            fire_time = 10000000 + random.random()

        G.nodes[src_node]['event_id'] = event_id
        new_time = global_clock + fire_time
        return new_time, src_node, new_state, event_id
