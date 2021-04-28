import numpy as np
import networkx as nx
import random


def f(states, s, ind):
    f = 1 + ind * ((states == 0) * s) + (1 - ind) * ((states == 1) * s)
    #f = 1 + (states * s)
    return f


def sim_bd(N, s, a, b):
    G = nx.complete_graph(N)
    state = dict(G.nodes)
    N = len(state)

    # Initialize state with resident (zeros)
    state[0] = 1
    for idx in range(1, N):
        state[idx] = 0

    n_current = 0
    n_end = 0
    ind = random.randint(0, 1)
    while (np.sum(list(state.values())) != 0) and (np.sum(list(state.values())) != N):

        if n_current == n_end:
            n_end = int(round(np.random.gamma(a, b)) + 1)
            n_current = 0
            ind = 1 - ind

        tmp_states = np.array(list(state.values()))

        weights = f(tmp_states, s, ind)
        node_sel = random.choices(list(state.keys()), weights=weights)

        nn = list(G.neighbors(node_sel[0]))
        nn_sel = random.choice(nn)

        state[nn_sel] = state[node_sel[0]]

        n_current += 1

    return (np.sum(list(state.values())) == N) * 1

def pfix_bd(N, s, a, b, iter):
    fix_sum = 0
    for i in range(iter):
        out = sim_bd(N, s, a, b)
        fix_sum += out
    pfix = fix_sum / float(iter)
    return pfix

# N: Population Size
# s: Selection coeficient (-1 < s < Inf | s > 0 is beneficial mutation and s < 0 is delitarious mutation)
# iter: number of Monte Carlo iterations
# a_list, b_list: list of parameters for Gamma Dist.
def analysis(N, s, iter, a_list, b_list):
    out = np.zeros([len(a_list), len(b_list)])
    i = 0
    for a in a_list:
        j = 0
        for b in b_list:
            pfix = pfix_bd(N, s, a, b, iter)
            out[i, j] = pfix

            j += 1
        i += 1
    np.savetxt("out.txt", out)

