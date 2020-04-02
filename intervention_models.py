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
import collections
import heapq
import copy


########################################################
# Superclass for any intervention
########################################################

class Intervention:
    def __init__(self):
        pass

    # current event is none if triggered by inrervention event
    def perform_intervention(self, G, model, last_event, global_clock, time_point_samples, event_queue, node_counter):
        # rewrite G inplace!
        pass


########################################################
# Source Tracing for Hill model
########################################################

class HillSourceTracing(Intervention):
    """
    A simple source tracing for the `CoronaHill` model. This is to be used in
    conjunction with the `CoronaHillWSourceTracing` model. A contact of a
    quarantined individual `I1Q`, `I2Q`, or `I3Q` is quarantined for a fixed
    amount of time (a `Q` is appended to its state)
    """

    def __init__(self, contact_detection_prob):
        assert 0 <= contact_detection_prob <= 1
        self.contact_detection_prob = contact_detection_prob
        # a list to track nodes that caused intervention
        self.interventions_done = list()

    def perform_intervention(self, G, model, last_event, global_clock, time_point_samples, event_queue, node_counter):
        if last_event is None:
            # TODO lift quarantine
            return

        # destruct event
        src_node, new_state, event_id = event_content
        if event_type != 'model':
            return
        new_time, event_type, event_content = current_event

        if not new_state.endswith("Q"):
            return

        # check if the intervention was performed already for this node
        if src_node in interventions_done:
            return
        interventions_done.append(src_node)

        for neighbor in G.neighbors(src_node):
            if np.random.rand() < self.contact_detection_prob:
                neighbor_state = G.nodes[neighbor]['state']
                if not neighbor_state.endswith('Q') or neighbor_state == 'D':
                    G.nodes[neighbor]['state'] = neighbor_state.append('Q')


########################################################
# Test Intervention
########################################################

class RandomRecover(Intervention):
    def perform_intervention(self, G, model, last_event, global_clock, time_point_samples, event_queue, node_counter):
        assert ('R' in model.states())
        random_node = random.choice([n for n in G.nodes()])
        old_state = G.nodes[random_node]['state']
        G.nodes[random_node]['state'] = 'R'
        G.nodes[random_node]['last_changed'] = global_clock
        node_counter[old_state] -= 1
        node_counter['R'] += 1


class RandomRewire(Intervention):
    def __init__(self, random_rewire_probability=1.0):
        self.random_rewire_probability = random_rewire_probability

    def perform_intervention(self, G, model, last_event, global_clock, time_point_samples, event_queue, node_counter):
        if random.random() < self.random_rewire_probability:
            edges = list(G.edges())
            while True:
                e1 = random.choice(edges)
                e2 = random.choice(edges)
                if len(set(list(e1) + list(e2))) == 4:  # make sure they do not share nodes
                    e1_list = list(e1)
                    e2_list = list(e2)
                    random.shuffle(
                        e1_list)  # more suble trick to avoid bias (dont rewire primarily lower nodes and higher nodes with each other)
                    random.shuffle(e2_list)
                    new_edge1 = (e1_list[0], e2_list[0])
                    new_edge2 = (e1_list[1], e2_list[1])
                    if new_edge1 in edges or new_edge2 in edges:  # make sure edges do not exists already
                        continue
                    G.add_edge(*new_edge1)
                    G.add_edge(*new_edge2)
                    G.remove_edge(*e1)
                    G.remove_edge(*e2)

                    for rewired_node in set(list(e1) + list(e2)):
                        e = model.next_event(G, rewired_node, global_clock)
                        heapq.heappush(event_queue, e)
                    break
