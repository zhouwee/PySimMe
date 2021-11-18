import time
import numpy as np
import collections


class PrettyTable():
    '''
    For format output the simplex table
    '''
    def __init__(self):
        # all element is str
        self.table = []
        return None
    
    def add_row(self, row):
        self.table.append(row)
        return None
    
    def pretty(self, hlines=[], vlines=[], col_width='c'):

        n_row, n_col = len(self.table), len(self.table[0])
        for i, e in enumerate(hlines):
            if e < 0: hlines[i] = n_row + e - 1
        for i, e in enumerate(vlines):
            if e < 0: vlines[i] = n_col + e - 1
        
        # column width
        col_width = [0 for j in range(n_col)]
        for row in self.table:
            for j, e in enumerate(row):
                col_width[j] = max(col_width[j], len(e))
        if col_width in ['c', 'center']:
            col_width = np.array(col_width)
            col_width[1:-1] = np.max(col_width[1:-1])
        elif col_width in ['e', 'each']:
            pass
        elif col_width in ['a', 'all']:
            col_width[:] = np.max(col_width)

        # extra char
        extra_width = n_col + 5

        doub_line = '=' * (extra_width + np.sum(col_width))
        sing_line = '-' * (extra_width + np.sum(col_width))

        # head line 
        cont = doub_line + '\n'
        for i, row in enumerate(self.table): 
            cont_row = ' '
            for j, e in enumerate(row):
                cont_row = cont_row + e.rjust(col_width[j]) + ' '
                # vertical lines
                if j in vlines: cont_row = cont_row + '| '
            cont = cont + cont_row + '\n'
            # horizontal lines
            if i in hlines: cont = cont + sing_line + '\n' 
        # bottom line
        cont = cont + doub_line + '\n'
        return cont




class RevisedSimplexMethod():
    '''
    The revised simplex method to solve the LP
    '''
    def __init__(self, c, A, b, c_0=0., var_names=None, mini=True):
        '''

        Args:
        ---
        A： nested list or ndarray, shape of (m, n)
            constraint coefficient matrix
        c:  list or ndarray, shape of (n,)
            object coefficient vector
        b:  list or ndarray, shape of (m,)
            right-hand side vector
        var_names:  list, its length should equal to the number of decision variables (n)
                    decision variables' names
        mini:   bool
                mini=True, is to minimize the object function
                mini=False, is to maximize the object function
        '''
        self.c = np.array(c, dtype=np.float_)
        self.A = np.array(A, dtype=np.float_)
        self.b = np.array(b, dtype=np.float_)
        self.c_0 = -c_0
        self.mini = mini
        # the number of constraints
        self.m = self.A.shape[0]
        # the number of decision variables
        self.n = self.A.shape[1]
        # decision variables' names
        self.var_names = var_names
        if self.var_names is None:
            self.var_names = ['x'+str(i) for i in range(1, self.n+1)]
        # indicate the whether is a two-phase simplex method
        self.phase = False # True: two-phase
        
        # check matrix A
        self.check(self.A)
        # the statue of solution
        self.statue = None

        # logs: for procedure
        # number of iteration
        self.n_iter = 0
        self.logs = {}
        self.logs['phase'] = []
        self.logs['ent_var'] = []
        self.logs['lev_var'] = []
        self.logs['basis'] = []
        self.logs['non_basis'] = []
        self.logs['w'] = []
        self.logs['c'] = []
        self.logs['A'] = []
        self.logs['b'] = []
        self.logs['c_0'] = []
        return None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def check(self, A):
        '''
        Check matrix A is fulfill the conditions of Standard Form of LP

        Args:
        -----
        A:  numpy.ndarray, shape of (m,n)

        '''
        (_m, _n) = A.shape
        try:
            assert _m <= _n
        except AssertionError:
            print('The number of decision variables (n={0}) should not least than the number of constraints (m={1})!'.format(_n, _m))
            raise
        
        try:
            assert np.linalg.matrix_rank(A) == _m
        except AssertionError:
            print('Redundant constraint exists! (Rank(A)={0} < m={1})'.format(np.linalg.matrix_rank(A), _m))
            raise
        return None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def record_logs(self, **kwargs):
        '''
        record some useful logs

        Args:
        -----
        **kwargs:

        '''
        for key, value in kwargs.items():
            self.logs[key].append(value)
        return None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def init_basis(self, A):
        '''
        From the constraint coefficient matrix, find a BFS
        Maybe return not a full basis

        Args:
        -----
        A:  numpy.ndarray, shape of (m,n)

        Return:
        -------
        basis:

        non_basis:

        '''
        (_m, _n) = A.shape
        basis = []
        for i in range(_n):
            r = collections.Counter(A[:, i].tolist())
            if (r[1.] == 1) and (r[0.] == _m-1):
                if not(i in basis):
                    basis.append(i)
        
        # get the non-basic variable
        non_basis = list(range(_n))
        for e in basis: non_basis.remove(e)

        return basis, non_basis
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def find_ent_var(self, c_D, non_basis, mini):
        '''
        To find an entering variable x_r from non_basis

        Args:
        ---
        c_D:    ndarray, shape of (n,)
                current object coefficient vector of non-basis
        non_basis:  list, it's length equal to n-m
                    the index of non-basic variables
        mini:   bool
                To find the most negative element (mini=True) for minimum LP
                or the most positive element (mini=True) for maximum LP
        '''
        if mini:
            index = np.argmin(c_D) 
        else:
            index = np.argmax(c_D) 
        return non_basis[index]
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def find_lev_var(self, b, p_s, basis):
        '''
        To find leaving variable from basis
        To find the less ratio (b_i/p_is) of p_s for all p_is>0.
        
        Args:
        ---
        b:  ndarray, shape of (m,)
            the current right-hand vector  
        p_s:    ndarray, shape of (m,)
                the current constrain coefficient vector of the entering variable x_r    
        basis:  list, it's length equal to m
                the index of basic variables
        '''
        _m = p_s.shape[0]
        lev_var = None
        _ratio = np.inf
        for i in range(_m):
            if p_s[i] <= 0.:
                continue
            else:
                ratio = b[i]/p_s[i]
                if ratio < _ratio:
                    lev_var = basis[i]
                    _ratio = ratio
        return lev_var
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def find_art_lev_var(self, c_D, non_basis):
        '''
        To find an entering variable x_r from non_basis and non-artifical variables
        Specially using for Phase I, it must be a minimal object, which indicates to find a most negative cost coefficient

        Args:
        ---
        c:    ndarray, shape of (n,)
                current object coefficient vector of non-basis
        non_basis:  list
                    the index of non-basic variables
        
        '''
        ent_c = 0.
        ent_var = None
        for i, e in enumerate(non_basis):
            if e < self.n:  # is a non-artifical variable
                if c_D[i] < ent_c:
                    ent_c = c_D[i]
                    ent_var = e
        return ent_var
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def init_phase1(self, c, A, b):
        '''
        Introduce artifical variables for two-phase simplex method in phase I:
        The standard form of Phase I:
         len:  n   m   1   1   |len              n+m+1   1
             | x | s | z | b   |               |   x   | b | 
         -w  | 0 | 1 | 0 |     | 1         -w  |   0   | 0 | 1
         -z  | c | 0 | 1 | c_0 | 1    =>   x_B |   A   | b | m+1
         x_B | A | I | 0 | b   | m          
        
        Args:
        -----
        A:  ndarray, shape of (m,n)
            original constraint coefficient
        '''
        (m, n) = A.shape 
        # names of artifical variables
        self.art_names = ['S'+str(i) for i in range(1, m+1)]

        # basic variable for Phase I
        ph1_basis = list(range(n, n+m))
        # get the non-basic variable
        ph1_non_basis = list(range(n+m))
        for e in ph1_basis:
            ph1_non_basis.remove(e)

        # the cost coefficient of w(x) for Phase I
        ph1_w = np.zeros(n+m, dtype=np.float_)
        ph1_w[n:] = 1
        ph1_c = np.concatenate((self.c, np.zeros(m, dtype=np.float_)))
        ph1_A = np.block([A, np.identity(m, dtype=np.float_)])
        ph1_b = b
        return ph1_w, ph1_c, ph1_A, ph1_b, ph1_basis, ph1_non_basis
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def iteration(self, c, A, b, basis, mini, phase):
        '''
        One iteration of simplex method, where the basis have be found

        Args:
        -----
        c:
        
        A:
        
        b:

        basis:
        
        mini:

        phase:

        '''
        (m, n) = A.shape 
        # get the non-basic variable
        non_basis = list(range(n))
        for e in basis: non_basis.remove(e)

        # Step 0: prepare B, B_inv, D, c_D, c_B, b, c_0
        B, D = A[:, basis], A[:, non_basis]
        c_B, c_D = c[basis], c[non_basis] 
        B_inv = np.linalg.inv(B)
        x_B = np.dot(B_inv, b)
        b = np.dot(B_inv, b)
        c_0 = np.dot(c_B, c_B)

        # Step 1：calculate current c_D
        c_D = (c_D.T - np.dot(np.dot(c_B.T, B_inv), D)).T 

        # Step 2: Determine the entering variable
        enter_var = self.find_ent_var(c_D, non_basis, mini=mini)
           
        # calculate the corresponding p_s
        p_s = np.dot(B_inv, A[:, enter_var])

        # Step 3: from non-basis variables find a leaving variable x_r
        leave_var = self.find_lev_var(b, p_s, basis)
        if leave_var is None: # no variable should leave basis
            self.statue = 2 # unbounded solution
            return enter_var, leave_var, basis, non_basis, c_D

        # Step 4: Updata basis and non-basis
        basis[basis.index(leave_var)] = enter_var
        basis.sort()
        non_basis[non_basis.index(enter_var)] = leave_var
        non_basis.sort()
        return enter_var, leave_var, basis, non_basis, c_D
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def compute(self):
        '''
        Iteration procedures of revised simplex method
        '''
        # record the running time
        start_time = time.process_time()
        # intial basis 
        basis, non_basis = self.init_basis(self.A)
        c, A, b = self.c, self.A, self.b 

        if len(basis) < self.m:
            self.phase = True            
            ph1_w, ph1_c, ph1_A, ph1_b, ph1_basis, ph1_non_basis = self.init_phase1(c, A, b)
            self.ph1_w, self.ph1_c, self.ph1_A, self.ph1_b, self.ph1_basis = ph1_w, ph1_c, ph1_A, ph1_b, ph1_basis
            # Phase I:
            while np.any(np.array(ph1_basis) >= self.n): # stop until all non-artifical variable leave the basis
                # print(ph1_basis, ph1_non_basis, self.n)
                # record logs
                self.n_iter = self.n_iter + 1
                self.record_logs(phase=True, basis=ph1_basis.copy(), non_basis=ph1_non_basis.copy())
                # each iteration
                ph1_enter_var, ph1_leave_var, ph1_basis, ph1_non_basis, c_D = self.iteration(ph1_w, ph1_A, ph1_b, ph1_basis, mini=True, phase=True)
                self.record_logs(ent_var=ph1_enter_var, lev_var=ph1_leave_var)
            # Phase I terminated:
            ph1_c_B = ph1_c[ph1_basis]
            ph1_B_inv = np.linalg.inv(ph1_A[:, ph1_basis])
            ph1_c = (self.ph1_c.T - np.dot(np.dot(ph1_c_B.T, ph1_B_inv), ph1_A)).T
            ph1_A = np.dot(ph1_B_inv, ph1_A)
            ph1_b = np.dot(ph1_B_inv, ph1_b)
            # self.ph1_c_0 = self.c_0 - np.dot(np.dot(ph1_c_B.T, ph1_B_inv), self.ph1_b)
            # print(ph1_basis, ph1_b, ph1_c[ph1_basis])
            self.ph1_c_0 = self.c_0 - np.dot(ph1_c[ph1_basis], ph1_b)
            c = ph1_c[:self.n]
            A = ph1_A[:, :self.n]
            b = ph1_b
            basis = ph1_basis.copy()

        # Phase II:
        while True:
            # record logs
            self.n_iter = self.n_iter + 1
            self.record_logs(phase=False, basis=basis.copy(), non_basis=non_basis.copy())
            # each iteration
            enter_var, leave_var, basis, non_basis, c_D = self.iteration(c, A, b, basis, mini=self.mini, phase=False)
            self.record_logs(ent_var=enter_var, lev_var=leave_var)
            # stop criterion
            if self.mini:
                if np.all(c_D >= 0.):
                    self.statue = 0 # successfully find the optimal solution 
                    break
            else:
                if np.all(c_D <= 0.):
                    self.statue = 0 
                    break
            # 
            if leave_var is None:
                break
        
        # record the running time
        self.run_time = time.process_time() - start_time
        # the optimal BFS
        self.opt_basis = self.logs['basis'][-1]
        self.opt_non_basis = self.logs['non_basis'][-1]
        return None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def get_optimal(self):
        '''
        obtain the optimal solutions and objective function value

        Return:
        -------
        sol:    ndarray, shape of (n, )
                The optimal solutions of the input LP
        obj:    float,
                The optimal objective function value
        '''
        x_B = np.dot(np.linalg.inv(self.A[:, self.opt_basis]), self.b)
        sol = np.zeros(self.n, dtype=np.float_)
        sol[self.opt_basis] = x_B
        obj = np.dot(sol, self.c)
        return sol, obj
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def get_table_logs(self):
        '''
        Get the logs of simplex table
        '''
        for i, bfs in enumerate(self.logs['basis']):
            if self.logs['phase'][i]:
                ph1_bfs = bfs
                ph1_c_B = self.ph1_c[ph1_bfs]
                ph1_B_inv = np.linalg.inv(self.ph1_A[:, ph1_bfs])
                ph1_x_B = np.dot(ph1_B_inv, self.ph1_b)
                ph1_w = (self.ph1_w.T - np.dot(np.dot(self.ph1_w[ph1_bfs].T, ph1_B_inv), self.ph1_A)).T
                ph1_c = (self.ph1_c.T - np.dot(np.dot(ph1_c_B.T, ph1_B_inv), self.ph1_A)).T
                ph1_A = np.dot(ph1_B_inv, self.ph1_A)
                ph1_b = np.dot(ph1_B_inv, self.ph1_b)
                ph1_c_0 = self.c_0 - np.dot(np.dot(ph1_c_B.T, ph1_B_inv), self.ph1_b)
                # record logs about phase I
                self.logs['w'].append(ph1_w)
                self.logs['c'].append(ph1_c)
                self.logs['A'].append(ph1_A)
                self.logs['b'].append(ph1_b)
                self.logs['c_0'].append(ph1_c_0)
            else:
                c_B = self.c[bfs]
                B_inv = np.linalg.inv(self.A[:, bfs])
                x_B = np.dot(B_inv, self.b)
                # record logs about phase II
                self.logs['c'].append((self.c.T - np.dot(np.dot(c_B.T, B_inv), self.A)).T)
                self.logs['A'].append(np.dot(B_inv, self.A))
                self.logs['b'].append(np.dot(B_inv, self.b))
                if self.phase:
                    self.logs['c_0'].append(self.ph1_c_0 - np.dot(np.dot(c_B.T, B_inv), self.b))
                else:
                    self.logs['c_0'].append(self.c_0 - np.dot(np.dot(c_B.T, B_inv), self.b))
        return None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def sensitivity_analysis(self):
        '''
        Sensitivity Analysis of cost coefficient and right-hand vector
        '''
        bfs, nbfs = self.opt_basis, self.opt_non_basis
        B, D = self.A[:, bfs], self.A[:, nbfs]
        B_inv = np.linalg.inv(self.A[:, bfs])
        c_B, c_D = self.c[bfs], self.c[nbfs]
        b = self.b

        self.range_cc = {}
        self.range_cc['lb'] = np.repeat(-np.inf, self.n)
        self.range_cc['ub'] = np.repeat(np.inf, self.n)

        for nb in self.opt_non_basis:
            if self.mini:
                self.range_cc['lb'][nb] = np.dot(np.dot(c_B.T, B_inv), self.A[:, nb])
            else:
                self.range_cc['ub'][nb] = np.dot(np.dot(c_B.T, B_inv), self.A[:, nb])

        for i, _bs in enumerate(self.opt_basis):
            lhs = np.dot(B_inv[i, :], D)
            c_B_i = np.delete(c_B, i)
            B_inv_i = np.delete(B_inv, i, axis=0)
            rhs = self.c[nbfs] - np.dot(np.dot(c_B_i.T, B_inv_i), D)
            lb_ix = lhs < 0.
            ub_ix = lhs > 0.
            try:
                if self.mini:
                    self.range_cc['lb'][_bs] = np.max(rhs[lb_ix] / lhs[lb_ix])
                    self.range_cc['ub'][_bs] = np.min(rhs[ub_ix] / lhs[ub_ix])
                else:
                    self.range_cc['lb'][_bs] = np.max(rhs[ub_ix] / lhs[ub_ix])
                    self.range_cc['ub'][_bs] = np.min(rhs[lb_ix] / lhs[lb_ix])
            except:
                continue

        self.range_b = {}
        self.range_b['lb'] = np.repeat(-np.inf, self.m)
        self.range_b['ub'] = np.repeat(np.inf, self.m)

        for i, _b_i in enumerate(b.tolist()):
            lhs = B_inv[:, i]
            rhs = -np.dot(np.delete(B_inv, i, axis=1), np.delete(b, i))
            lb_ix = lhs > 0.
            ub_ix = lhs < 0.
            try:
                self.range_b['lb'][i] = np.max(rhs[lb_ix] / lhs[lb_ix])
                self.range_b['ub'][i] = np.min(rhs[ub_ix] / lhs[ub_ix])
            except:
                continue

        return None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def cont_sensitivity_analysis(self):
        '''
        output sensitivity analysis
        '''
        self.sensitivity_analysis()
        range_cc = self.range_cc.copy()
        range_b = self.range_b.copy()
        range_cc['lb'] = list(map(lambda x: str(round(x, self.dec)), range_cc['lb'].tolist()))
        range_cc['ub'] = list(map(lambda x: str(round(x, self.dec)), range_cc['ub'].tolist()))
        range_b['lb']  = list(map(lambda x: str(round(x, self.dec)), range_b['lb'].tolist()))
        range_b['ub']  = list(map(lambda x: str(round(x, self.dec)), range_b['ub'].tolist()))

        cc_table = PrettyTable()
        cc_table.add_row(['', 'lower bound', 'upper bound'])
        for i in range(self.n):
            cc_table.add_row([self.var_names[i], range_cc['lb'][i], range_cc['ub'][i]])
        cont_cc = cc_table.pretty(hlines=[0], col_width='a')

        b_table = PrettyTable()
        b_table.add_row(['', 'lower bound', 'upper bound'])
        for j in range(self.m):
            b_table.add_row(['b'+str(j+1), range_b['lb'][j], range_b['ub'][j]])
        cont_b = b_table.pretty(hlines=[0], col_width='a')

        cont = 'Part 2: Sensitivity Analysis: \n \n' 
        cont = cont + 'Sensitivity analysis of cost coefficients c: \n'
        cont = cont + cont_cc + '\n'
        cont = cont + 'Sensitivity analysis of right-hand side vector b: \n'
        cont = cont + cont_b + '\n'
        return cont
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    def cont_basic_information(self):
        '''
        output the basic information about the solved LP 
        '''
        obj_type = 'Minimization' if self.mini else 'Maximization'
        basis = self.logs['basis'][0]
        sol, obj = self.get_optimal()
        sol = list(map(lambda x: str(round(x, self.dec)), sol))
        obj = str(round(obj, self.dec))
        cont = 'Part 1: Overall Information: \n\n'
        if self.phase: 
            cont = cont + 'Fail to find a basic feasible solution, the two-phase simplex method is used.'
        else:
            cont = cont + 'Successfully find a initial basic feasible solution: '
            cont = cont + ', '.join([self.var_names[basis[i]] for i in range(len(basis))])
        cont = cont + '\n\n'
        cont = cont + 'Object function type: ' + obj_type + '\n'
        cont = cont + 'Optimal solution: ' + ', '.join(sol) + '\n'
        cont = cont + 'Optimal object function: ' + obj + '\n'
        cont = cont + 'Total iterations: ' + str(self.n_iter) + '\n'
        cont = cont + 'Run time: ' + str(round(self.run_time, self.dec))
        return cont
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def display_table(self, w, c, A, b, c_0, basis, phase):
        '''
        formatted output the simplex table
        '''
        dec = self.dec
        var_names_exp = self.var_names_exp.copy()
        n_row, n_col = A.shape
        if phase: w = np.around(w, decimals=dec).tolist(); w = list(map(str, w))
        c = np.around(c, decimals=dec).tolist(); c = list(map(str, c))
        A = np.around(A, decimals=dec).tolist(); A = [list(map(str, i)) for i in A]
        b = np.around(b, decimals=dec).tolist(); b = list(map(str, b))
        c_0 = str(np.around(c_0, decimals=dec))

        tab = PrettyTable()

        row=['']; row.extend(self.var_names); 
        if phase: row.extend(self.art_names)
        row.append('')
        tab.add_row(row)  

        if phase: 
            row=['-w']; row.extend(w); row.append('')
            tab.add_row(row)  

        row=['-z']; row.extend(c); row.append(c_0)
        tab.add_row(row)  
        for i in range(n_row):
            row=[var_names_exp[basis[i]]]; row.extend(A[i]); row.append(b[i])
            tab.add_row(row)
        
        if phase:
            cont = tab.pretty(hlines=[0,2], vlines=[0,-1])
        else:
            cont = tab.pretty(hlines=[0,1], vlines=[0,-1])
        return cont
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def cont_simplex_table(self):
        '''
        output each iteration information
        '''
        dec = self.dec
        cont = 'Part 3: Detailed Simplex Tables for Each Iteration: \n \n'

        self.var_names_exp = self.var_names.copy()
        if self.phase: self.var_names_exp.extend(self.art_names)

        for iter in range(self.n_iter):
            if self.phase:
                str_phase = 'I' if self.logs['phase'][iter] else 'II'
                str_head = 'Phase {0:<2}, Iteration {1}: '.format(str_phase, iter+1)
            else:
                str_head = 'Iteration {0}: '.format(iter+1)

            try:
                if self.phase and not(self.logs['phase'][iter+1]):
                    str_var = 'Phase I terminated. \n + Find the BFS for Phase II are '
                    str_var = str_var + ', '.join([self.var_names[i] for i in self.logs['basis'][iter]])
            except:
                pass

            if iter == self.n_iter-1:  # the last iteration
                str_var = 'Iteration terminated.'
            else:
                str_var = 'Entering variable: {0}, Leaving variable: {1}'.format(
                    self.var_names_exp[self.logs['ent_var'][iter]],
                    self.var_names_exp[self.logs['lev_var'][iter]],
                )
            
            if self.logs['phase'][iter]:
                table = self.display_table(
                    w = self.logs['w'][iter],
                    c = self.logs['c'][iter], 
                    A = self.logs['A'][iter],
                    b = self.logs['b'][iter], 
                    c_0 = self.logs['c_0'][iter],
                    basis = self.logs['basis'][iter],
                    phase = self.logs['phase'][iter]
                )
            else:
                table = self.display_table(
                    w = None,
                    c = self.logs['c'][iter], 
                    A = self.logs['A'][iter],
                    b = self.logs['b'][iter], 
                    c_0 = self.logs['c_0'][iter],
                    basis = self.logs['basis'][iter],
                    phase = self.logs['phase'][iter]
                )
            cont = cont + str_head + '\n' + str_var + '\n'
            cont = cont + table + '\n'
        return cont
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def summary(self, outfile=None, dec=3):
        '''
        output and 
        '''
        self.dec = dec
        self.get_table_logs()
        cont = self.cont_basic_information() + '\n\n'
        cont = cont + self.cont_sensitivity_analysis()
        cont = cont + self.cont_simplex_table() + '\n\n'
        print(cont)
        if not(outfile is None):
            with open(outfile, 'w') as f:
                f.write(cont)
        return None
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-