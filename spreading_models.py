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
# Superclass for any spreading model
########################################################

class SpreadingModel:
    # you probably do not want to touch this class
    def __init__(self, number_of_units=1):
        self.number_of_units = number_of_units  # only relevant for deterministic solution
        pass

    def states(self):
        return ['I', 'S']

    def get_number_of_units(self):
        try:
            return self.number_of_units
        except:
            return 1.0

    def colors(self):
        palette = sns.color_palette("muted", len(self.states()))
        colors = {s: list(palette[i]) for i, s in enumerate(self.states())}
        return colors

    def get_init_labeling(self, G):
        return {n: random.choice(self.states()) for n in range(G.number_of_nodes())}

    # wrapper (logic in generate event)
    # this method is the important one in the super-class, do not overwrite it
    def next_event(self, G, src_node, global_clock):
        event_id = G.nodes[src_node]['event_id']
        event_id += 1
        new_time, new_state = self.generate_event(G, src_node, global_clock)
        G.nodes[src_node]['event_id'] = event_id

        # build event
        event_type = 'model'
        event_content = (src_node, new_state, event_id)
        event = (new_time, event_type, event_content)
        return event

    def generate_event(self, G, src_node, global_clock):
        return global_clock + random.random(), random.choice(self.states())

    def aggregate(self, node_state_counts):
        return node_state_counts

    # for the deterministic solution
    def ode_init(self):
        raise NotImplementedError
        # return [(1.0/len(self.states()))*self.get_number_of_units() for _ in self.states()]

    def ode_func(self, population_vector, t):
        raise NotImplementedError
        # return [0.001*i for i in range(len(self.states()))]


########################################################
# Classical SIS Model
########################################################

class SISmodel(SpreadingModel):
    def __init__(self, infection_rate):
        self.infection_rate = infection_rate

    def states(self):
        return ['I', 'S']

    def get_init_labeling(self, G):
        init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def colors(self):
        return {'S': sns.xkcd_rgb['denim blue'], 'I': sns.xkcd_rgb['pinkish red']}

    def generate_event(self, G, src_node, global_clock):
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

        new_time = global_clock + fire_time
        return new_time, new_state


########################################################
# Classical SIR Model
########################################################

class SIRmodel(SpreadingModel):
    def __init__(self, infection_rate):
        self.infection_rate = infection_rate

    def states(self):
        return ['I', 'S', 'R']

    def colors(self):
        return {'S': sns.xkcd_rgb['denim blue'], 'I': sns.xkcd_rgb['pinkish red'], 'R': sns.xkcd_rgb['medium green']}

    def get_init_labeling(self, G):
        init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def generate_event(self, G, src_node, global_clock):
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

        new_time = global_clock + fire_time
        return new_time, new_state


########################################################
# Corona Model (inspired by Alison Hill)
########################################################

class CoronaHill(SpreadingModel):
    # find the excellent online tool at: https://alhill.shinyapps.io/COVID19seir/
    # conversion to a networked model based on scaling infection rate based on the mean degree of the network

    def __init__(self, scale_by_mean_degree=True, init_exposed=None, number_of_units=1, scale_inf_rate=1):

        b1 = 0.500  # / number of nodes      # infection rate from i1
        b2 = 0.100  # / number of nodes      # infection rate from i2
        b3 = 0.100  # / number of nodes      # infection rate from i3
        a = 0.200  # e to i1
        g1 = 0.133  # i1 to r
        g2 = 0.125  # i2 to r
        g3 = 0.075  # i3 to r
        p1 = 0.033  # i1 to i2
        p2 = 0.042  # i2 to i3
        u = 0.050  # i3 to death

        self.s_to_e_dueto_i1 = b1 * scale_inf_rate
        self.s_to_e_dueto_i2 = b2 * scale_inf_rate
        self.s_to_e_dueto_i3 = b3 * scale_inf_rate
        self.e_to_i1 = a
        self.i1_to_i2 = p1
        self.i2_to_i3 = p2
        self.i3_to_d = u
        self.i1_to_r = g1
        self.i2_to_r = g2
        self.i3_to_r = g3
        self.scale_by_mean_degree = scale_by_mean_degree
        self.init_exposed = init_exposed

        self.number_of_units = number_of_units  # only relevant for deterministic ODE

    def states(self):
        return ['S', 'E', 'I1', 'I2', 'I3', 'R', 'D']

    def colors(self):
        colors = {'S': sns.xkcd_rgb['denim blue'], 'E': sns.xkcd_rgb['bright orange'], 'I1': sns.xkcd_rgb['light red'],
                  'I2': sns.xkcd_rgb['pinkish red'], 'I3': sns.xkcd_rgb['deep pink'], 'R': sns.xkcd_rgb['medium green'],
                  'D': sns.xkcd_rgb['black']}
        colors['I_total'] = 'gray'  # need to add states from finalize
        return colors

    def get_init_labeling(self, G):
        if self.init_exposed is not None:
            init_node_state = {n: 'S' for n in range(G.number_of_nodes())}
            for exp_node in self.init_exposed:
                init_node_state[exp_node] = 'E'
            return init_node_state
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

    def generate_event(self, G, src_node, global_clock):
        if G.nodes[src_node]['state'] == 'S':
            new_state = 'E'
            neighbors = G.neighbors(src_node)
            count_i1 = len([n for n in neighbors if G.nodes[n]['state'] == 'I1'])
            count_i2 = len([n for n in neighbors if G.nodes[n]['state'] == 'I2'])
            count_i3 = len([n for n in neighbors if G.nodes[n]['state'] == 'I3'])
            if count_i1 + count_i2 + count_i3 == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = count_i1 * self.s_to_e_dueto_i1 + count_i2 * self.s_to_e_dueto_i2 + count_i3 * self.s_to_e_dueto_i3
                if self.scale_by_mean_degree:
                    mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
                    node_rate /= mean_degree
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
            assert (False)

        new_time = global_clock + fire_time
        return new_time, new_state

    # ODE

    # has to be a vector in the order of models.states()
    def ode_init(self):
        init = [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        init = [x * self.number_of_units for x in init]
        return init

    def ode_func(self, population_vector, t):
        s = population_vector[0]
        e = population_vector[1]
        i1 = population_vector[2]
        i2 = population_vector[3]
        i3 = population_vector[4]
        r = population_vector[5]
        d = population_vector[6]

        s_grad = -(
                self.s_to_e_dueto_i1 / self.number_of_units * i1 + self.s_to_e_dueto_i2 / self.number_of_units * i3 + self.s_to_e_dueto_i3 / self.number_of_units * i3) * s
        e_grad = (
                         self.s_to_e_dueto_i1 / self.number_of_units * i1 + self.s_to_e_dueto_i2 / self.number_of_units * i3 + self.s_to_e_dueto_i3 / self.number_of_units * i3) * s - self.e_to_i1 * e
        i1_grad = self.e_to_i1 * e - (self.i1_to_r + self.i1_to_i2) * i1
        i2_grad = self.i1_to_i2 * i1 - (self.i2_to_r + self.i2_to_i3) * i2
        i3_grad = self.i2_to_i3 * i2 - (self.i3_to_r + self.i3_to_d) * i3
        r_grad = self.i1_to_r * i1 + self.i2_to_r * i2 + self.i3_to_r * i3
        d_grad = self.i3_to_d * i3

        grad = [s_grad, e_grad, i1_grad, i2_grad, i3_grad, r_grad, d_grad]

        return grad


# Model by José Lourenço et al.
# (not tested, no deads yet)
# Oxford model: https://www.medrxiv.org/content/10.1101/2020.03.24.20042291v1.full.pdf
class CoronaLourenco(SpreadingModel):
    def __init__(self, scale_by_mean_degree=True, number_of_units=1):

        self.sigma = 1.0 / 4.5  # recovery rate
        self.r_0 = 2.75
        self.beta = self.sigma * self.r_0  # infection rate

        self.scale_by_mean_degree = scale_by_mean_degree
        self.number_of_units = number_of_units

    def states(self):
        return ['I', 'S', 'R']

    def colors(self):
        return {'S': sns.xkcd_rgb['denim blue'], 'I': sns.xkcd_rgb['pinkish red'], 'R': sns.xkcd_rgb['medium green']}

    def get_init_labeling(self, G):
        init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def generate_event(self, G, src_node, global_clock):
        if G.nodes[src_node]['state'] == 'I':
            new_state = 'R'
            recovery_rate = self.sigma
            fire_time = -np.log(random.random()) / recovery_rate
        elif G.nodes[src_node]['state'] == 'S':
            new_state = 'I'
            inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] == 'I'])
            if inf_neighbors == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = inf_neighbors * self.beta
                if self.scale_by_mean_degree:
                    mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
                    node_rate /= mean_degree
                fire_time = -np.log(random.random()) / node_rate
        else:
            new_state = 'R'
            fire_time = 10000000 + random.random()

        new_time = global_clock + fire_time
        return new_time, new_state

    # has to be a vector in the order of models.states()
    def ode_init(self):
        init = [0.05, 0.95, 0.0]
        init = [x * self.number_of_units for x in init]
        return init

    def ode_func(self, population_vector, t):
        i = population_vector[0]
        s = population_vector[1]
        r = population_vector[2]

        s_grad = -(self.beta / self.number_of_units * i) * s
        i_grad = (self.beta / self.number_of_units * i) * s - (self.sigma * i)
        r_grad = self.sigma * i

        grad = [i_grad, s_grad, r_grad]

        return grad


class CoronaBase(SpreadingModel):
    def __init__(self, scale_by_mean_degree=True, number_of_units=1):

        self.s_to_e_dueto_im = 0.5
        self.s_to_e_dueto_imq = 0.1
        self.s_to_e_dueto_is = 0.5
        self.s_to_e_dueto_isq = 0.1
        self.s_to_e_dueto_ish = 0.05

        self.e_to_im = 0.2 * 4 / 5
        self.e_to_is = 0.2 * 1 / 5

        self.im_to_imq = 1
        self.im_to_r = 1 / 5

        self.imq_to_r = 1 / 5

        self.is_to_isq = 1 / 2
        self.is_to_ish = 1 / 4
        self.is_to_r = 1 / 7
        self.is_to_d = 1 / 14

        self.isq_to_ish = 1 / 2
        self.isq_to_r = 1 / 7
        self.isq_to_d = 1 / 7

        self.number_of_units = number_of_units
        self.scale_by_mean_degree = scale_by_mean_degree

    def markov_firing(self, state_to_rate):
        state_rate = list(state_to_rate.items())
        state_firetime = [(s, -np.log(random.random()) / r) for s, r in state_rate]
        state_firetime = sorted(state_firetime, key=lambda s_t: s_t[1])
        min_state = state_firetime[0][0]
        min_time = state_firetime[0][1]
        return min_state, min_time

    def states(self):
        return ['S', 'E', 'Im', 'Imq', 'Is', 'Isq', 'Ish', 'R', 'D']

    def generate_event(self, G, src_node, global_clock):
        local_state = G.nodes[src_node]['state']

        if local_state == 'S':
            new_state = 'E'
            neighbors = G.neighbors(src_node)
            count_im = len([n for n in neighbors if G.nodes[n]['state'] == 'Im'])
            count_imq = len([n for n in neighbors if G.nodes[n]['state'] == 'Imq'])
            count_is = len([n for n in neighbors if G.nodes[n]['state'] == 'Is'])
            count_isq = len([n for n in neighbors if G.nodes[n]['state'] == 'Isq'])
            count_ish = len([n for n in neighbors if G.nodes[n]['state'] == 'Ish'])

            if count_im + count_imq + count_is + count_isq + count_ish == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = count_im * self.s_to_e_dueto_im + count_imq * self.s_to_e_dueto_imq + count_is * self.s_to_e_dueto_is + count_isq * self.s_to_e_dueto_isq + count_ish * self.s_to_e_dueto_ish
                if self.scale_by_mean_degree:
                    mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
                    node_rate /= mean_degree
                fire_time = -np.log(random.random()) / node_rate

        elif local_state == 'E':
            state_to_rate = {'Im': self.e_to_im, 'Is': self.e_to_is}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'Im':
            state_to_rate = {'Imq': self.im_to_imq, 'R': self.im_to_r}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'Imq':
            fire_time = -np.log(random.random()) / self.imq_to_r
            new_state = 'R'

        elif local_state == 'Is':
            state_to_rate = {'Isq': self.is_to_isq, 'Ish': self.is_to_ish, 'R': self.is_to_r, 'D': self.is_to_d}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'Isq':
            state_to_rate = {'Ish': self.isq_to_ish, 'R': self.isq_to_r, 'D': self.isq_to_d}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'Ish':
            state_to_rate = {'R': self.ish_to_r, 'D': self.ish_to_d}
            new_state, fire_time = self.markov_firing(state_to_rate)

        else:
            new_state = local_state
            fire_time = 1000000 + random.random()

        new_time = global_clock + fire_time
        return new_time, new_state
