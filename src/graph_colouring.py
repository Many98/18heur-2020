import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib

import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from heur_graph import GeneticOptimization, UniformMultipoint, GraphMutation
from objfun_graph import Graph

NUM_RUNS = 100
maxeval = 200

g = Graph()



def graph_colouring(graph, maxeval, num_runs, N, M, Tsel1, Tsel2, mutation, crossover):
    results = []
    heur_name = 'GO_{}'.format(N)
    n_colors = graph.init_chromatic_number

    for n in tqdm(range(num_runs), 'Testing {}'.format(heur_name)):
        while True:
            

            result = GeneticOptimization(graph, maxeval, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, 
                                        mutation=mutation, crossover=crossover).search(n_colors)
            result['run'] = n
            result['heur'] = heur_name
            result['N'] = N
            result['color'] = None
            results.append(result)
                
            
            if result['best_y'] != 0:
                print(f"Graph is {n_colors+1} colorable")
                result['color'] = n_colors + 1
                n_colors = graph.init_chromatic_number
                break
            else:
                n_colors -= 1
                print(f'Found feasible coloring for {n_colors+1} colors')
    return pd.DataFrame(results, columns=['heur', 'run', 'N', 'color', 'best_x', 'best_y', 'neval'])

if __name__ == '__main__':
    mutation = GraphMutation()
    crossover = UniformMultipoint(1)

    out = graph_colouring(g, maxeval, NUM_RUNS, 200, 400, 1, 0.1, mutation, crossover)
    print('done')