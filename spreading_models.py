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


class Corona:
    def __init__(self):
        self.s_to_e = 0.5
        self.e_to_i1 = 0.5
        self.i1_to_i2 = 0.2
        self.i2_to_i3 = 0.2
        self.i3_to_d = 0.5
        self.i1_to_r = 0.1
        self.i2_to_r = 0.05
        self.i3_to_r = 0.05


    def states(self):
        return ['S', 'E', 'I1', 'I2', 'I3', 'R', 'D']

    def get_init_labeling(self, G):
        init_node_state = {n: ('E' if random.random() > 0.90 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state


    def aggregate(self, node_state_counts):
        node_state_counts['I_total'] = [0 for _ in range(len(node_state_counts['I1']))]
        for i, v in enumerate(node_state_counts['I1']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['I2']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['I3']):
            node_state_counts['I_total'][i] += v
        return node_state_counts


    def next_event(self, G, src_node, global_clock):
        event_id = G.nodes[src_node]['event_id']
        event_id += 1

        if G.nodes[src_node]['state'] == 'S':
            new_state = 'E'
            inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] in ['I1','I2','I3']])
            if inf_neighbors == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = inf_neighbors * self.s_to_e
                fire_time = -np.log(random.random()) / node_rate

        elif G.nodes[src_node]['state'] == 'E':
            new_state = 'I1'
            fire_time = -np.log(random.random()) / self.e_to_i1

        elif G.nodes[src_node]['state'] == 'I1':
            new_state_c1 = 'I2'
            fire_time_c1 = -np.log(random.random()) / self.i1_to_i2
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i1_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2

        elif G.nodes[src_node]['state'] == 'I2':
            new_state_c1 = 'I3'
            fire_time_c1 = -np.log(random.random()) / self.i2_to_i3
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i2_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2

        elif G.nodes[src_node]['state'] == 'I3':
            new_state_c1 = 'D'
            fire_time_c1 = -np.log(random.random()) / self.i3_to_d
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i3_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2
        elif G.nodes[src_node]['state'] == 'R':
            new_state = 'R'
            fire_time = 10000000 + random.random()
        elif G.nodes[src_node]['state'] == 'D':
            new_state = 'D'
            fire_time = 10000000 + random.random()
        else:
            print('no matching state')
            assert(False)

        G.nodes[src_node]['event_id'] = event_id
        new_time = global_clock + fire_time
        return new_time, src_node, new_state, event_id