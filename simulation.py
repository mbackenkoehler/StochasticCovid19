import warnings
warnings.filterwarnings("ignore")

import matplotlib
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
from visualization import viz_simulate, geom_graph


#
# Config
#

from spreading_models import *


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


def simulation_run(G, model, time_point_samples, at_leat_one=False, max_steps=None, node_wise_matrix=None):
    global_clock = 0.0
    step_i = 0
    time_point_sample_index = 0
    event_queue = [] # init
    x_values = list(time_point_samples)
    y_values = {state: list() for state in model.states()}  # record of trajectory
    node_counter = {state: len([n for n in G.nodes() if G.nodes[n]['state'] == state]) for state in model.states()}
    if node_wise_matrix is not None:
        assert(node_wise_matrix.shape == (G.number_of_nodes(), len(x_values)))
    # init event_id
    for node_i in G.nodes():
        G.nodes[node_i]['event_id'] = 0

    # init queue
    for node_i in G.nodes():
        e = model.next_event(G, node_i, global_clock)
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
            for state in model.states():
                y_values[state].append(node_counter[state])
            if node_wise_matrix is not None:
                for n in G.nodes():
                    node_wise_matrix[n, time_point_sample_index] = model.states().index(G.nodes[n]['state'])
            x_values = x_values[1:]
            time_point_sample_index += 1
        if len(x_values) == 0:
            return y_values

        # reject
        if G.nodes[src_node]['event_id'] != event_id:
            e = model.next_event(G, src_node, global_clock)
            heapq.heappush(event_queue, e)
            continue
        if at_leat_one and 'I' in model.states() and node_counter['I'] == 1:
            e = model.next_event(G, src_node, global_clock)
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
        e = model.next_event(G, src_node, global_clock)
        heapq.heappush(event_queue, e)
        for neighbor in G.neighbors(src_node):
            e = model.next_event(G, neighbor, global_clock)
            heapq.heappush(event_queue, e)

    return y_values




def simulate(G, model, time_point_samples, num_runs=30, outpath = 'output.pdf', max_steps=None, node_wise_matrix=None):
    G = nx.convert_node_labels_to_integers(G)
    init_node_state = model.get_init_labeling(G)

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

        node_state_counts = simulation_run(G_run_i, model, time_point_samples, at_leat_one=False, max_steps=max_steps, node_wise_matrix=node_wise_matrix)

        try:
            node_state_counts = model.aggregate(node_state_counts)
        except:
            pass
        print('.', end='')
        for x_i, time_point in enumerate(time_point_samples):
            for node_state, fractions in node_state_counts.items():
                run_id_column.append(run_i)
                time_point_column.append(time_point)
                state_column.append(node_state)
                fraction_column.append(node_state_counts[node_state][x_i]/G.number_of_nodes())
    print('finished simulations')

    df = pd.DataFrame({'run_id': run_id_column, 'Time': time_point_column, 'State': state_column, 'Fraction':fraction_column})
    df.to_csv(outpath.replace('.pdf','.csv'))
    lineplot(df, model, time_point_samples, outpath)
    return df

def lineplot(df, model, time_point_samples, outpath):
    plt.clf()
    palette = None
    try:
        palette = model.get_colors()
    except:
        pass
    sns.lineplot(x="Time", y="Fraction", hue='State', data=df, ci=95, palette = palette)
    plt.ylim([0,1])
    plt.xlim([0, time_point_samples[-1]])
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig(outpath, bbox_inches="tight")
    plt.show(block=False)

def final_mean_in_state(df, state='R'):
    last_time_point = np.max(df['Time'])
    df = df[df.apply(lambda line: line['Time'] == last_time_point and line['State'] == state, axis=1)]
    if len(df['Fraction']) == 0:
        return 0.0
    return np.mean(df['Fraction'])

def final_mean(df, model):
    return {state: final_mean_in_state(df, state=state) for state in model.states()}


def visualization(G, model, time_point_samples, outpath = 'vit_out.gif', node_pos=None):
    G = nx.convert_node_labels_to_integers(G)
    node_wise_matrix = np.zeros([G.number_of_nodes(), len(time_point_samples)])
    simulate(G, model, time_point_samples, num_runs=1, outpath = outpath, node_wise_matrix=node_wise_matrix)
    viz_simulate(G, time_point_samples, node_wise_matrix, model, node_pos=node_pos)



if __name__ == "__main__":
    #cv = get_critical_value(G)
    #sis_model = SISmodel(infection_rate=cv*3)
    #sir_model = SIRmodel(infection_rate=cv * 7)

    G, node_pos = geom_graph()
    # G = nx.grid_2d_graph(20, 20)
    # Note that visualization is super slow currently
    # To reduce gif size you might want to use "gifsicle -i output_simulation_movie.gif -O3 --colors 100 -o anim-opt.gif"
    visualization(G, Corona(init_exposed=[0], scale_by_mean_degree=False), np.linspace(0,120,60), outpath='output_singlerun_geom_viz.pdf', node_pos=node_pos)

    corona_model = Corona()
    time_point_samples =  np.linspace(0,100,100)
    df = simulate(nx.grid_2d_graph(10,10), corona_model, time_point_samples, outpath = 'output_grid.pdf')
    print('final mean grid:', final_mean(df, corona_model))
    df = simulate(nx.complete_graph(100), corona_model, time_point_samples, outpath='output_complete.pdf')
    print('final mean complete:', final_mean(df, corona_model))
    df = simulate(nx.erdos_renyi_graph(n=100, p=0.1), corona_model, time_point_samples, outpath='output_erdosrenyi.pdf')
    print('final mean erdos renyi:', final_mean(df, corona_model))
