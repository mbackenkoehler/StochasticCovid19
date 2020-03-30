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
from tqdm import tqdm

#
# Config
#

from spreading_models import *
from intervention_models import *

RUN_NUM = 10
if 'TRAVIS' in os.environ:  # dont put too much load on travis (only relevant for testing)
    RUN_NUM = 3


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


def simulation_run(G, model, time_point_samples, at_leat_one=False, max_steps=None, node_wise_matrix=None, interventions=None):
    global_clock = 0.0
    step_i = 0
    time_point_sample_index = 0
    event_queue = []  # init
    x_values = list(time_point_samples)
    y_values = {state: list() for state in model.states()}  # record of trajectory
    node_counter = {state: len([n for n in G.nodes() if G.nodes[n]['state'] == state]) for state in model.states()}
    if node_wise_matrix is not None:
        assert (node_wise_matrix.shape == (G.number_of_nodes(), len(x_values)))
    if interventions is not None and (not hasattr(interventions, '__iter__')):
        interventions = [interventions]

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
        current_event = heapq.heappop(event_queue)
        new_time, src_node, new_state, event_id = current_event
        global_clock = new_time

        # store
        while len(x_values) > 0 and global_clock >= x_values[0]:
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

        # create new event
        e = model.next_event(G, src_node, global_clock)
        heapq.heappush(event_queue, e)
        for neighbor in G.neighbors(src_node):
            e_neig = model.next_event(G, neighbor, global_clock)
            heapq.heappush(event_queue, e_neig)

        # perform interventions
        if interventions is not None:
            for intervention in interventions:
                intervention.perform_intervention(G, model, current_event, global_clock, time_point_samples, event_queue, node_counter)

    return y_values


def simulate(G, model, time_point_samples, num_runs=None, outpath='output/output.pdf', max_steps=None,
             node_wise_matrix=None, interventions=None):
    if num_runs is None:
        num_runs = RUN_NUM
    G = nx.convert_node_labels_to_integers(G)
    init_node_state = model.get_init_labeling(G)
    pbar = tqdm(total=num_runs)
    pbar.set_description('Simulations')

    for node in G.nodes():
        G.nodes[node]['state'] = init_node_state[node]
        G.nodes[node]['last_changed'] = 0.0

    # create data frame to store values
    run_id_column = list()
    time_point_column = list()
    state_column = list()
    fraction_column = list()
    for run_i in range(num_runs):
        G_run_i = copy.deepcopy(G)  # to not overwrite

        node_state_counts = simulation_run(G_run_i, model, time_point_samples, at_leat_one=False, max_steps=max_steps,
                                           node_wise_matrix=node_wise_matrix, interventions=interventions)

        try:
            node_state_counts = model.aggregate(node_state_counts)
        except:
            pass
        pbar.update(1)
        for x_i, time_point in enumerate(time_point_samples):
            for node_state, fractions in node_state_counts.items():
                run_id_column.append(run_i)
                time_point_column.append(time_point)
                state_column.append(node_state)
                fraction_column.append(node_state_counts[node_state][x_i] / G.number_of_nodes())
    pbar.close()

    df = pd.DataFrame(
        {'run_id': run_id_column, 'Time': time_point_column, 'State': state_column, 'Fraction': fraction_column})
    df.to_csv(outpath + '.csv')
    lineplot(df, model, time_point_samples, outpath)
    return df


def lineplot(df, model, time_point_samples, outpath):
    plt.clf()
    palette = None
    try:
        palette = model.colors()
    except:
        pass
    sns.lineplot(x="Time", y="Fraction", hue='State', data=df, ci=95, palette=palette)
    plt.ylim([0, 1])
    plt.xlim([0, time_point_samples[-1]])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(outpath, bbox_inches="tight")
    plt.show(block=False)
    # plt.ylim([0,0.15])
    # plt.savefig(outpath.replace('.','trunc.'), bbox_inches="tight")
    # plt.show(block=False)


def final_mean_in_state(df, state='R'):
    last_time_point = np.max(df['Time'])
    df = df[df.apply(lambda line: line['Time'] == last_time_point and line['State'] == state, axis=1)]
    if len(df['Fraction']) == 0:
        return 0.0
    return np.mean(df['Fraction'])


def final_mean(df, model):
    return {state: final_mean_in_state(df, state=state) for state in model.states()}


def visualization(G, model, time_point_samples, outpath='vit_out.gif', node_pos=None):
    G = nx.convert_node_labels_to_integers(G)
    node_wise_matrix = np.zeros([G.number_of_nodes(), len(time_point_samples)])
    simulate(G, model, time_point_samples, num_runs=1, outpath=outpath, node_wise_matrix=node_wise_matrix)
    viz_simulate(G, time_point_samples, node_wise_matrix, model, node_pos=node_pos)


def solve_ode(model, time_point_samples, outpath='output_gif/output_ode.pdf'):
    plt.clf()
    from scipy.integrate import odeint
    init = model.ode_init()
    f = model.ode_func
    sol = odeint(f, init, time_point_samples)
    np.savetxt(outpath + '.csv', sol)
    for state_i, state in enumerate(model.states()):
        sol_i = sol[:, state_i]
        try:
            c = model.colors()[state]
        except:
            c = None
        plt.plot(time_point_samples, sol_i, label=state, c=c, alpha=0.8, lw=2)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylim([0, model.get_number_of_units()])
    plt.xlim([0, time_point_samples[-1]])
    plt.savefig(outpath, bbox_inches="tight")
    plt.show(block=False)
    print('final values of ODE: ', {model.states()[i]: sol[-1, i] for i in range(len(model.states()))})
    return sol

if __name__ == "__main__":
    os.system('mkdir output/')
    time_point_samples = np.linspace(0, 100, 100)

    #
    # Test SIS Model
    #
    G = nx.grid_2d_graph(5, 5)
    cv = get_critical_value(G)  # find interesting infection rate
    sis_model = SISmodel(infection_rate=1.0)
    # solve_ode(sis_model,  time_point_samples, outpath = 'output/output_ode_sis.pdf')   #not implemented
    df = simulate(G, sis_model, time_point_samples, outpath='output/output_sis_grid.pdf')
    print('final mean sis grid:', final_mean(df, sis_model))

    #
    # Test SIR Model
    #
    sir_model = SIRmodel(infection_rate=0.5)
    # solve_ode(sir_model, time_point_samples, outpath = 'output/output_ode_sir.pdf')  # not implemented
    df = simulate(nx.grid_2d_graph(5, 5), sir_model, time_point_samples, outpath='output/output_sir_grid.pdf')
    print('final mean sir grid:', final_mean(df, sir_model))

    #
    # Test Corona Model on Grid Network
    #
    corona_model = CoronaHill()
    solve_ode(corona_model, time_point_samples, outpath='output/output_ode_corona.pdf')
    df = simulate(nx.grid_2d_graph(10, 10), corona_model, time_point_samples, outpath='output/output_corona_grid.pdf')
    print('final mean grid:', final_mean(df, corona_model))

    #
    # Test Corona Model on Geometric Network
    #
    G_geom, node_pos = geom_graph()
    corona_model = CoronaHill()
    df = simulate(G_geom, corona_model, time_point_samples, outpath='output/output_corona_geom.pdf')
    print('final mean geom:', final_mean(df, corona_model))

    #
    # Test SuperClass (gives nonsense data)
    #
    spr_model = SpreadingModel()
    df = simulate(nx.grid_2d_graph(3, 3), spr_model, time_point_samples, outpath='output/output_superclass_grid.pdf')
    print('final mean superclass grid:', final_mean(df, spr_model))
    # solve_ode(spr_model, time_point_samples, outpath = 'output/output_ode_superclass.pdf')

    #
    # Create Gif with Corona Model
    #
    # Note that visualization is super slow currently
    # To reduce gif size you might want to use "gifsicle -i output_simulation_movie.gif -O3 --colors 100 -o anim-opt.gif"
    os.system('mkdir output_gif/')
    G_geom, node_pos = geom_graph()
    visualization(G_geom, CoronaHill(init_exposed=[0], scale_by_mean_degree=False), np.linspace(0, 120, 60),
                  outpath='output_gif/output_singlerun_geom_viz.pdf', node_pos=node_pos)

    #
    # Create Gif with SIS Model
    #
    # Note that visualization is super slow currently
    # To reduce gif size you might want to use "gifsicle -i output_simulation_movie.gif -O3 --colors 100 -o anim-opt.gif"
    G_geom, node_pos = geom_graph(node_num=100)
    visualization(G_geom, SISmodel(infection_rate=0.5), np.linspace(0, 20, 60),
                  outpath='output_gif/output_singlerun_geom_sis_viz.pdf', node_pos=node_pos)

    #
    # Test Corona Model on Complete Network
    #
    corona_model = CoronaHill()
    df = simulate(nx.complete_graph(100), corona_model, time_point_samples, outpath='output/output_corona_complete.pdf')
    print('final mean complete:', final_mean(df, corona_model))

    #
    # Test Corona Model on Erdos Renyi Network
    #
    corona_model = CoronaHill()
    df = simulate(nx.erdos_renyi_graph(n=100, p=0.05), corona_model, time_point_samples,
                  outpath='output/output_corona_erdosrenyi.pdf')
    print('final mean erdosrenyi:', final_mean(df, corona_model))

    #
    # Test SIR Model with random reovery intervention
    #
    sir_model = SIRmodel(infection_rate=1.3)
    df = simulate(nx.grid_2d_graph(10, 10), sir_model, time_point_samples, outpath='output/output_sir_grid_wointervent.pdf', num_runs = 100)
    print('final mean sir grid:', final_mean(df, sir_model))
    sir_model = SIRmodel(infection_rate=1.3)
    rec_inv = RandomRecover()
    df = simulate(nx.grid_2d_graph(10, 10), sir_model, time_point_samples, outpath='output/output_sir_grid_withintervent.pdf', interventions=rec_inv, num_runs = 100)
    print('final mean sir grid random recovery:', final_mean(df, sir_model))