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


#
# Config
#

class SISmodel:
    def __init__(self, infection_rate = 0.5):
        self.infection_rate = infection_rate

    def states(self):
        return ['I', 'S']

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



NODE_STATES = ['I', 'S']

# the next event function specifies the model dynamics
def next_event(G, inf_rate, src_node, global_clock):
    event_id = G.nodes[src_node]['event_id']
    event_id += 1

    if  G.nodes[src_node]['state'] == 'I':
        new_state = 'S'
        fire_time = -np.log(random.random()) # recov-rate is alsways 1
    else:
        new_state = 'I'
        inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] == 'I'])
        if inf_neighbors == 0:
            fire_time = 10000000 + random.random()
        else:
            node_rate = inf_neighbors * inf_rate
            fire_time = -np.log(random.random()) / node_rate


    G.nodes[src_node]['event_id'] = event_id
    new_time = global_clock + fire_time
    return new_time, src_node, new_state, event_id



#
# Simulation Code
#

def get_critical_value(contact_network):
    A = nx.adj_matrix(contact_network)
    A = A.toarray()
    E = scipy.linalg.eigvals(A)
    E = sorted(list(E))
    l = E[-1].real
    beta = 1.0 / l
    return beta


def simulation_run(G, effective_infection_rate, time_point_samples, at_leat_one=False, max_steps=None):
    global_clock = 0.0
    step_i = 0
    event_queue = [] # init
    x_values = list(time_point_samples)
    y_values = {state: list() for state in NODE_STATES}  # record of trajectory
    node_counter = {state: len([n for n in G.nodes() if G.nodes[n]['state'] == state]) for state in NODE_STATES}
    # init event_id
    for node_i in G.nodes():
        G.nodes[node_i]['event_id'] = 0

    # init queue
    for node_i in G.nodes():
        e = next_event(G, effective_infection_rate, node_i, global_clock)
        heapq.heappush(event_queue, e)


    while len(x_values) > 0:

        if max_steps is not None and step_i >= max_steps:
            print('Abort simulation')
            return y_values

        # pop next event
        new_time, src_node, new_state, event_id = heapq.heappop(event_queue)
        global_clock = new_time

        # store
        while len(x_values)>0 and global_clock >= x_values[0]:
            for state in NODE_STATES:
                y_values[state].append(node_counter[state])
            x_values = x_values[1:]
        if len(x_values) == 0:
            return y_values

        # reject
        if G.nodes[src_node]['event_id'] != event_id:
            e = next_event(G, effective_infection_rate, src_node, global_clock)
            heapq.heappush(event_queue, e)
            continue
        if at_leat_one and 'I' in NODE_STATES and node_counter['I'] == 1:
            e = next_event(G, effective_infection_rate, src_node, global_clock)
            heapq.heappush(event_queue, e)
            continue

        # apply event
        old_state = G.nodes[src_node]['state']
        G.nodes[src_node]['last_changed'] = global_clock
        node_counter[old_state] -= 1
        G.nodes[src_node]['state'] = new_state
        node_counter[new_state] += 1

        step_i += 1

        #create new event
        e = next_event(G, effective_infection_rate, src_node, global_clock)
        heapq.heappush(event_queue, e)
        for neighbor in G.neighbors(src_node):
            e = next_event(G, effective_infection_rate, neighbor, global_clock)
            heapq.heappush(event_queue, e)

    return y_values




def simulate(G, effective_infection_rate, time_point_samples, num_runs=500, outpath = 'simu_out.pdf', init_node_state = None, max_steps=None):
    G = nx.convert_node_labels_to_integers(G)
    if init_node_state is None:
        init_node_state = {n: random.choice(NODE_STATES) for n in G.nodes()}

    for node in G.nodes():
        G.nodes[node]['state'] = init_node_state[node]
        G.nodes[node]['last_changed'] = 0.0

    #create data frame to store values
    run_id_column = list()
    time_point_column = list()
    state_column = list()
    fraction_column = list()
    for run_i in range(num_runs):
        G_run_i = copy.deepcopy(G) # to not overwrite
        node_state_counts = simulation_run(G_run_i, effective_infection_rate, time_point_samples, at_leat_one=False, max_steps=max_steps)
        print('.', end='')
        for x_i, time_point in enumerate(time_point_samples):
            for node_state, fractions in node_state_counts.items():
                run_id_column.append(run_i)
                time_point_column.append(time_point)
                state_column.append(node_state)
                fraction_column.append(node_state_counts[node_state][x_i]/G.number_of_nodes())
    print('')


    df = pd.DataFrame({'run_id': run_id_column, 'Time': time_point_column, 'State': state_column, 'Fraction':fraction_column})
    df.to_csv(outpath.replace('.pdf','.csv'))
    sns.lineplot(x="Time", y="Fraction", hue='State', data=df, ci=95)
    plt.ylim([0,1])
    plt.savefig(outpath)



if __name__ == "__main__":
    plt.clf()
    G = nx.grid_2d_graph(10,10)
    cv = get_critical_value(G)*3
    time_point_samples =  np.linspace(0,10,30)
    init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
    simulate(G, cv, time_point_samples, init_node_state=init_node_state)