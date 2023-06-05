from objfun import ObjFun
import numpy as np
from numba import jit
import multiprocessing as mp
from random import sample



class Graph(ObjFun):

    """
    graph representation
    """

    def __init__(self, filepath='/home/emanuel/Documents/Matematika/HEUR/queen5_5.col', random=False, n_vertices=50):
        """
        """

        # https://mat.gsia.cmu.edu/COLOR04/INSTANCES/queen5_5.col
        if random:
            pass
        else:
            with open(filepath) as f:
                lines = f.readlines()
            e = [i.replace('e', '').replace('\n', '').split() for i in lines]
            self.edges = np.array(e[4:]).astype(int)
            self.n_vertices = 25
            self.n_edges = self.edges.shape[0]
            self.chromatic_num = 5  # known chromatic number of this particular graph

            self.adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype=int)
            self.adjacency_matrix[self.edges[:, 0] - 1, self.edges[:, 1] - 1] = 1
            self.adjacency_matrix[self.edges[:, 1] - 1, self.edges[:, 0] - 1] = 1

        self.init_chromatic_number = self.adjacency_matrix.sum(0).max() + 1
        super().__init__(fstar=0, a=np.zeros(self.n_vertices, dtype=int), b=np.ones(self.n_vertices, dtype=int))

    def __str__(self) -> str:
        return str(self.adjacency_matrix)

    def generate_point(self, *args):
        """
        generate random coloring
        :return: random coloring vector
        """
        return np.random.randint(1, args[0]+1, self.n_vertices, dtype=int)
    
    def generate_population(self, size, n_colors):
        """
        generate population of random coloring vectors i.e. matrix where every column is one element of population (candidate colouring)
        :return: random coloring matrix
        """
        np.random.randint(1, n_colors+1, (self.n_vertices, size), dtype=int)

    @jit
    def get_neighborhood(self, x, n_colors, restricted=True, sample_size=1000):
        """
        Solution neighborhood generating function.
        We use simple 1-exchange neighbourhood
        :param x: point i.e. colouring (does not need to be feasible colouring)
        :param n_colours: number of colours
        :param restricted: whether to consider only vertices in conflicts
        :return: list of points in the neighborhood of the x
        """
        
        neighbours = []

        conflict_matrix = self.get_conflicts(x) if restricted else np.ones((self.n_vertices, self.n_vertices), dtype=int)

        for i in range(self.n_vertices):
            if conflict_matrix.sum(1)[i] > 0:
                for j in range(1, n_colors+1):
                    y = x.copy()
                    y[i] = j
                    if not (y == x).all():
                        neighbours.append(y)

        return sample(neighbours, sample_size)

    def get_conflicts(self, x):
        conflict_matrix = self.adjacency_matrix.copy()
        for i in range(self.n_vertices):
            for j in range(self.n_vertices):
                if x[i] != x[j]:
                    conflict_matrix[i, j] = 0
        return conflict_matrix
    
    def eval_neighbours(self, neighbours, parallel=False):
        if parallel:
            with mp.Pool(mp.cpu_count() - 2) as p:
                results = p.map(self.evaluate, neighbours)
        else:
            results = list(map(self.evaluate, neighbours))

        return results
    
    def evaluate(self, x):
        """
        Objective function evaluating function. Essentialy it returns number of 
        conflicting vertices of candidate solution `x`
        :return: objective function value
        """

        return self.get_conflicts(x).sum() // 2
    
if __name__ == '__main__':

    g = Graph()
    print(g)
