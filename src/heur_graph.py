import numpy as np
from heur import StopCriterion, Heuristic
from heur_aux import is_integer
from random import sample

class GraphMutation(object):
    """
    baseline mutation used in graph colouring problem
    """
    def __init__(self) -> None:
        pass

    def mutate(self, x, n_colors, proba=0.1):
        """
        randomly select vertex and change its color randomly
        """
        if np.random.uniform() <= proba:
            position = np.random.randint(0, x.shape[0])
            s = [i for i in range(1, n_colors+1) if i != x[position]]

            x[position] = sample(s, 1)[0]

        return x

class Crossover:
    """
    Baseline crossover  - randomly chooses "genes" from parents
    """

    def __init__(self):
        pass

    def crossover(self, x, y):
        z = np.array([x[i] if np.random.uniform() < 0.5 else y[i] for i in np.arange(x.size)], dtype=x.dtype)
        return z


class UniformMultipoint(Crossover):
    """
    Uniform n-point crossover
    """

    def __init__(self, n):
        self.n = n  # number of crossover points

    def crossover(self, x, y):
        co_n = self.n + 1
        n = np.size(x)
        z = x*0
        k = 0
        p = np.ceil(n/co_n).astype(int)
        for i in np.arange(1, co_n+1):
            ix_from = k
            ix_to = np.minimum(k+p, n)
            z[ix_from:ix_to] = x[ix_from:ix_to] if np.mod(i, 2) == 1 else y[ix_from:ix_to]
            k += p
        return z
    
class RandomExchange(Crossover):
    """
    Randomly combines parents
    """

    def __init__(self):
        pass

    def crossover(self, x, y):

        if is_integer(x):
            z = np.array([sample([x[i], y[i]], 1)[0] for i in np.arange(x.size)],
                         dtype=x.dtype)
            return z
        else:
            return x
        
class MinConflict(Crossover):
    """
    Greedy approach which takes takes gene from parent which have at particular gene less number of conflicts
    """

    def __init__(self, graph):
        self.g = graph

    def crossover(self, x, y):
        xx = self.g.get_conflicts(x).sum(0)
        yy = self.g.get_conflicts(y).sum(0)
        return np.array([x[i] if xx[i] < yy[i] else y[i] for i in range(x.shape[0])])



class RandomCombination(Crossover):
    """
    Randomly combines parents
    """

    def __init__(self):
        pass

    def crossover(self, x, y):

        if is_integer(x):
            z = np.array([np.random.randint(np.min([x[i], y[i]]), np.max([x[i], y[i]]) + 1) for i in np.arange(x.size)],
                         dtype=x.dtype)
        else:
            z = np.array([np.random.uniform(np.min([x[i], y[i]]), np.max([x[i], y[i]])) for i in np.arange(x.size)],
                         dtype=x.dtype)
        return z

class GeneticOptimization(Heuristic):

    def __init__(self, of, maxeval, N, M, Tsel1, Tsel2, mutation, crossover):

        Heuristic.__init__(self, of, maxeval)

        assert M > N, 'M should be larger than N'
        self.N = N  # population size
        self.M = M  # working population size
        self.Tsel1 = Tsel1  # first selection temperature
        self.Tsel2 = Tsel2  # second selection temperature
        self.mutation = mutation
        self.crossover = crossover
    
    @staticmethod
    def sort_pop(pop_x, pop_f):
        ixs = np.argsort(pop_f)
        pop_x = pop_x[ixs]
        pop_f = pop_f[ixs]
        return [pop_x, pop_f]

    @staticmethod
    def rank_select(temp, n_max):
        u = np.random.uniform(low=0.0, high=1.0, size=1)
        ix = np.minimum(np.ceil(-temp*np.log(u)), n_max)-1
        return ix.astype(int)
    
    def evaluate(self, x, type_):
        """
        Single evaluation of the objective function
        :param x: point to be evaluated
        :return: corresponding objective function value
        """
        x = [kk for kk in x]

        # paralelized evaluation of multiple candidate colourings (in this case population)
        y = self.of.eval_neighbours(x, type_=type_)
            
        self.neval += 1
        arg_best = np.argmin(y)
        best_y = y[arg_best]
        best_x = x[arg_best]
        if best_y < self.best_y:
            self.best_y = best_y
            self.best_x = best_x
        if self.of.get_fstar() == np.min(y):
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval == self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def search(self, n_colors, type_):

        try:
            
            # Initialization:
            pop_X = np.zeros([self.N, np.size(self.of.a)], dtype=self.of.a.dtype)  # population solution vectors
            pop_f = np.zeros(self.N)  # population fitness (objective) function values

            # a.) generate the population
            for i in np.arange(self.N):
                x = self.of.generate_point(n_colors)
                pop_X[i, :] = x
            
            # paralelized evaluation
            pop_f = np.array(self.evaluate(pop_X, type_))

            # b.) sort according to fitness function
            [pop_X, pop_f] = self.sort_pop(pop_X, pop_f)

            # Evolution iteration
            while True:
                # 1.) generate the working population
                work_pop_X = np.zeros([self.M, np.size(self.of.a)], dtype=self.of.a.dtype)
                work_pop_f = np.zeros(self.M)
                for i in np.arange(self.M):
                    parent_a_ix = self.rank_select(temp=self.Tsel1, n_max=self.N)  # select first parent
                    parent_b_ix = self.rank_select(temp=self.Tsel1, n_max=self.N)  # 2nd --//-- (not unique!)
                    par_a = pop_X[parent_a_ix, :][0]
                    par_b = pop_X[parent_b_ix, :][0]

                    # perform crossover
                    z = self.crossover.crossover(par_a, par_b)

                    # perform mutation
                    z_mut = self.mutation.mutate(z, n_colors, 0.4)

                    work_pop_X[i, :] = z_mut

                # paralelized evaluation
                work_pop_f = np.array(self.evaluate(work_pop_X, type_))

                # 2.) sort working population according to fitness function
                [work_pop_X, work_pop_f] = self.sort_pop(work_pop_X, work_pop_f)

                # 3.) select the new population
                ixs_not_selected = np.ones(self.M, dtype=bool)  # this mask will prevent us from selecting duplicates
                for i in np.arange(self.N):
                    sel_ix = self.rank_select(temp=self.Tsel2, n_max=np.sum(ixs_not_selected))
                    pop_X[i, :] = work_pop_X[ixs_not_selected][sel_ix, :]
                    pop_f[i] = work_pop_f[ixs_not_selected][sel_ix]
                    ixs_not_selected[sel_ix] = False

                # 4.) sort according to fitness function
                [pop_X, pop_f] = self.sort_pop(pop_X, pop_f)

        except StopCriterion:
            return self.report_end()
        except:
            raise

