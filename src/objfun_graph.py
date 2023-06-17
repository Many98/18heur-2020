from objfun import ObjFun
import numpy as np
#from numba import jit
import multiprocessing as mp
from random import sample



class Graph(ObjFun):

    """
    graph representation
    """

    def __init__(self, filepath, chromatic_number):
        """
        """

        # https://mat.gsia.cmu.edu/COLOR04/INSTANCES/queen5_5.col  # Xi = 5
        # https://mat.gsia.cmu.edu/COLOR04/  zeroin.i.1 # Xi = 49
        # https://cedric.cnam.fr/~porumbed/graphs/

        self.path = filepath
        self.chromatic_num = chromatic_number  # known chromatic number of this particular graph
        
        self.read_col()

        self.adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype=int)
        self.adjacency_matrix[self.edges[:, 0] - 1, self.edges[:, 1] - 1] = 1
        self.adjacency_matrix[self.edges[:, 1] - 1, self.edges[:, 0] - 1] = 1

        self.n_edges = self.adjacency_matrix.sum() // 2

        self.init_chromatic_number = self.adjacency_matrix.sum(0).max() + 1

        super().__init__(fstar=0, a=np.zeros(self.n_vertices, dtype=int), b=np.ones(self.n_vertices, dtype=int))

    def read_col(self):
        """
        read graph in .col format
        """
        with open(self.path) as f:
            lines = f.readlines()

        e = [i.replace('e', '').replace('\n', '').split() for i in lines if i.startswith('e') or i.startswith('p')]
        self.edges = np.array(e[1:]).astype(int)
    
        self.n_vertices = int(e[0][2])

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

    #@jit
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
        """
        get conflict matrix i.e. nxn matrix where n is number of vertices. 
        number 1 on (i,j) means that vertices i, j are in conflict i.e. they have same color.
        :param x: point i.e. colouring (does not need to be feasible colouring)
        :return: np.ndarray i.e. conflict matrix
        """
        conflict_matrix = self.adjacency_matrix.copy()
        for i in range(self.n_vertices):
            for j in range(self.n_vertices):
                if x[i] != x[j]:
                    conflict_matrix[i, j] = 0
        return conflict_matrix
    
    def eval_neighbours(self, neighbours, parallel=False, type_=0):
        """
        evaluates set of candidate colorings `neighbours`
        
        :param neighbours: list of points i.e. list of candidate colourings (does not need to be feasible colourings)
        :param parallel: whether to use parallelization (on process level)
        :param type_: type of objective function ... `type_=0` is classical objective func while `type_=1` assigns additional penalty
        :return: list of fitness values
        """
        if parallel:
            with mp.Pool(mp.cpu_count() - 2) as p:
                results = p.map(self.evaluate, neighbours)
        else:
            results = list(map(self.evaluate, neighbours, [type_ for i in range(len(neighbours))]))

        return results
    
    def evaluate(self, x, type=0):
        """
        Objective function evaluating function. Essentialy it returns number of 
        conflicting vertices of candidate solution `x`
        :param x: point i.e. candidate colouring (does not need to be feasible colouring)
        :param type_: type of objective function ... `type_=0` is classical objective func while `type_=1` assigns additional penalty
        :return: objective function value
        """
        if type == 0:
            return self.get_conflicts(x).sum() // 2
        elif type == 1:
            return self.get_conflicts(x).sum() // 2 - ((self.get_conflicts(x).sum(0) / self.adjacency_matrix.sum(0))).sum()
    
if __name__ == '__main__':

    zeroin = Graph('../zeroin.i.1.col', 49)
    queen5 = Graph('../queen5_5.col', 5)
    flat300_28 = Graph('../flat300_28_0.col.txt', 28)
    r250 = Graph('../r250.5.col.txt', 65)
    print(zeroin)
