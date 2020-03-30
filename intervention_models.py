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
# Dummy-Class for events (in the event queue)
# that trigger interventions
########################################################

class InterventionEvent:
    pass


########################################################
# Superclass for any intervention
########################################################

class Intervention:
    def __init__(self):
        pass

    def perform_intervention(self, G, model, last_event, global_clock, time_point_samples, heapq, node_counter):
        # rewrite G inplace!
        pass


########################################################
# Test Intervention
########################################################


class RandomRecover(Intervention):
    def perform_intervention(self, G, model, last_event, global_clock, time_point_samples, heapq, node_counter):
        assert('R' in model.states())
        random_node = random.choice([n for n in G.nodes()])
        old_state = G.nodes[random_node]['state']
        G.nodes[random_node]['state'] = 'R'
        G.nodes[random_node]['last_changed'] = global_clock
        node_counter[old_state] -= 1
        node_counter['R'] += 1



