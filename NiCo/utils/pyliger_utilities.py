import numpy as np
import scipy.sparse as sps
import numpy.linalg as nla
import scipy.linalg as sla
from numba import jit, njit, prange
#from ._utilities import _h5_idx_generator


def NMF_obj_eval(Xs, W, Hs, L, Sp, Vs=0):
    "Evaluates NMF objective function (sparsity, Vs optional)."
    if Vs == 0: Vs = [zeros((1, shape(Hs[i])[0])) for i in range(len(Xs))]
    obj = sum([np.linalg.norm(Xs[i] - (W+Vs[i]).dot(Hs[i]))**2
        for i in range(len(Xs))])
    if type(L) != list: L = [L]*len(Xs)
    pen = sum([L[i]*np.linalg.norm(Vs[i].dot(Hs[i]))**2 for i in range(len(Xs))])
    spars = 0
    if Sp != 0:
        spars = Sp*np.sum([np.sum(abs(Hs[i])) for i in range(len(Xs))])
    return obj+pen + spars




def iNMF(X, k,value_lambda=5.0,thresh=1e-6,max_iters=30,nrep=1,H_init=None,W_init=None,V_init=None,rand_seed=1,print_obj=False):
    #"Integrated NMF (sparsity optional)."
    #Ms = [Xs[i].shape[1] for i in range(K)]
    #W_f = np.zeros((N, D))
    #Hs_f = [np.zeros((D, Ms[i])) for i in range(K)]
    #Vs_f = [np.zeros((N, D)) for i in range(K)]
    # used from pyliger codes optimize_ALS _iNMF_ANLS.py
    N = len(X)
    num_genes  = X[0].shape[1]
    ns = [X[i].shape[0] for i in range(N)]

    best_obj = np.Inf
    for j in range(nrep):
        np.random.seed(seed=rand_seed + j - 1)

        ### 1. Initialization (W, V_i, H_i)
        W = np.abs(np.random.uniform(0, 2, (k, num_genes)))
        V = [np.abs(np.random.uniform(0, 2, (k, num_genes))) for i in range(N)]
        H = [np.abs(np.random.uniform(0, 2, (ns[i],k))) for i in range(N)]

        #print('1b',k,N,ns,W.shape)
        #print('2a',H[0].shape,H[1].shape,V[0].shape,V[1].shape)

        if W_init is not None:
            W = W_init

        if V_init is not None:
            V = V_init

        if H_init is not None:
            H = H_init

        delta = 1
        sqrt_lambda = np.sqrt(value_lambda)

        # Initial training obj
        obj_train_approximation = 0
        obj_train_penalty = 0
        for i in range(N):
            #print('maxi',X[i].shape,H[i].shape,W.shape,V[i].shape)
            obj_train_approximation += np.linalg.norm(X[i] - H[i] @ (W + V[i])) ** 2
            obj_train_penalty += np.linalg.norm(H[i] @ V[i]) ** 2

        obj0 = obj_train_approximation + value_lambda * obj_train_penalty

        ### 2. Iteration starts here
        for iter in range(max_iters):
            if delta > thresh:
                ## 1) update H matrix
                for i in range(N):
                    H[i] = nnlsm_blockpivot(A=np.hstack(((W + V[i]), sqrt_lambda * V[i])).transpose(),
                                            B=np.hstack((X[i], np.zeros((ns[i], num_genes)))).transpose())[0].transpose()

                ## 2) update V matrix
                for i in range(N):
                    V[i] = nnlsm_blockpivot(A=np.vstack((H[i], sqrt_lambda * H[i])),
                                            B=np.vstack(((X[i] - H[i] @ W), np.zeros((ns[i], num_genes)))))[0]

                ## 3) update W matrix
                W = nnlsm_blockpivot(A=np.vstack(H), B=np.vstack([(X[i] - H[i] @ V[i]) for i in range(N)]))[0]

                obj_train_prev = obj0
                obj_train_approximation = 0
                obj_train_penalty = 0
                for i in range(N):
                    obj_train_approximation += np.linalg.norm(X[i] - H[i] @ (W + V[i])) ** 2
                    obj_train_penalty += np.linalg.norm(H[i] @ V[i]) ** 2
                obj0 = obj_train_approximation + value_lambda * obj_train_penalty
                delta = np.absolute(obj_train_prev - obj0) / ((obj_train_prev + obj0) / 2)
            else:
                continue
            #print(iter,delta,obj0)

        if obj0 < best_obj:
            final_W = W
            final_H = H
            final_V = V
            best_obj = obj0
            best_seed = rand_seed + i - 1

        if print_obj:
            print('Objective: {}'.format(best_obj),iter,delta)

    #liger_object.W = final_W.transpose()

    ### 3. Save results into the liger_object
    #for i in range(N):
        #liger_object.adata_list[i].obsm['H'] = final_H[i]
        #liger_object.adata_list[i].varm['W'] = final_W.transpose()
        #liger_object.adata_list[i].varm['V'] = final_V[i].transpose()
        #idx = liger_object.adata_list[i].uns['var_gene_idx']
        #shape = liger_object.adata_list[i].shape
        #save_W = np.zeros((shape[1], k))
        #save_W[idx, :] = final_W.transpose()
        #save_V = np.zeros((shape[1], k))
        #save_V[idx, :] = final_V[i].transpose()
        #liger_object.adata_list[i].obsm['H'] = final_H[i]
        #liger_object.adata_list[i].varm['W'] = save_W
        #liger_object.adata_list[i].varm['V'] = save_V
    return final_H[0].T,final_H[1].T,final_W.T,final_V[0].T,final_V[1].T


def nonneg(x, eps=1e-16):
    """ Given a input matrix, set all negative values to be zero """
    x[x < eps] = eps
    return x


def _init_W(num_genes, k, rand_seed):
    """helper function to initialize a W matrix"""
    # set seed
    np.random.seed(seed=rand_seed)

    W = np.abs(np.random.uniform(0, 2, (num_genes, k)))

    # normalize columns of dictionaries
    W = W / np.sqrt(np.sum(np.square(W), axis=0))

    return W


def _init_V(num_cells, num_samples, k, Xs):
    """helper function to initialize a V matrix for in-memory mode"""
    # pick k sample from datasets as initial V matrix
    V = [Xs[i][:, np.random.choice(list(range(num_cells[i])), k)].toarray() for i in range(num_samples)]

    # normalize columns of dictionaries
    V = [V[i] / np.sqrt(np.sum(np.square(V[i]), axis=0)) for i in range(num_samples)]

    return V
"""
def _init_V_online(num_cells, num_samples, k, Xs, chunk_size, rand_seed):

    np.random.seed(seed=rand_seed)

    Vs = []
    for i in range(num_samples):
        # pick k sample from datasets as initial H matrix
        idx = np.sort(np.random.choice(list(range(num_cells[i])), k))
        V = []
        for left, right in _h5_idx_generator(chunk_size, num_cells[i]):
            select_idx = idx[(idx >= left) & (idx < right)] - left  # shift index because of handling chunk each time
            if select_idx.shape[0] > 0:  # only load chunks whose indexes are picked
                X = Xs[i]['scale_data'][left:right]
                V.append(X[select_idx, :])
        V = sps.vstack(V).transpose().toarray()

        # normalize columns of dictionaries
        V = V / np.sqrt(np.sum(np.square(V), axis=0))

        Vs.append(V)

    return Vs
"""

def _init_V_online(num_cell, k, X, chunk_size, rand_seed):
    """helper function to initialize a V matrix for online learning"""
    np.random.seed(seed=rand_seed)

    # pick k sample from datasets as initial H matrix
    idx = np.sort(np.random.choice(list(range(num_cell)), k))
    V = []
    for left, right in _h5_idx_generator(chunk_size, num_cell):
        select_idx = idx[(idx >= left) & (idx < right)] - left  # shift index because of handling chunk each time
        if select_idx.shape[0] > 0:  # only load chunks whose indexes are picked
            X_chunk = X['scale_data'][left:right]
            V.append(X_chunk[select_idx, :])
    V = sps.vstack(V).transpose().toarray()

    # normalize columns of dictionaries
    V = V / np.sqrt(np.sum(np.square(V), axis=0))

    return V


def _init_H(num_cells, num_samples, k):
    """helper function to initialize a H matrix"""
    H = [np.random.uniform(0, 2, (k, num_cells[i])) for i in range(num_samples)]
    return H

def _update_W_HALS(A, B, W, V):
    """helper function to update W matrix by HALS
    A = HiHi^t, B = XiHit, W = gene x k, V = [gene x k]"""
    for j in range(W.shape[1]):
        W_update_numerator = np.zeros(W.shape[0])
        W_update_denominator = 0.0
        for i in range(len(V)):
            W_update_numerator = W_update_numerator + B[i][:, j] - ((W + V[i]) @ A[i])[:, j]
            W_update_denominator += A[i][j, j]
        W[:, j] = nonneg(W[:, j] + W_update_numerator / W_update_denominator)

    return W


def _update_V_HALS(A, B, W, V, value_lambda):
    """helper function to update V matrix by HALS
    A = HiHi^t, B = XiHit, W = gene x k, V = [gene x k]"""
    for j in range(W.shape[1]):
        for i in range(len(V)):
            V[i][:, j] = nonneg(V[i][:, j] + (B[i][:, j] - (W + (1 + value_lambda) * V[i]) @ A[i][:, j]) / ((1 + value_lambda) * A[i][j, j]))

    return V


def _update_H_HALS(H, V, W, X, value_lambda):
    """helper function to update H matrix by HALS"""
    VitVi = [Vi.transpose() @ Vi for Vi in V]
    W_Vi = [W + Vi for Vi in V]
    W_Vi_sq = [W_Vii.transpose() @ W_Vii for W_Vii in W_Vi]
    for i in range(len(V)):
        for j in range(W.shape[1]):
            H[i][j, :] = nonneg(H[i][j, :] + (
                    W_Vi[i][:, j].transpose() @ X[i] - W_Vi[i][:, j].transpose() @ W_Vi[i] @ H[
                i] - value_lambda * VitVi[i][j, :] @ H[i]) / (
                                        W_Vi_sq[i][j, j] + value_lambda * VitVi[i][j, j]))
    return H


######use numba
"""
#@jit(nopython=True)
def _update_W_HALS(A, B, W, V):

    for j in range(W.shape[1]):
        W_update_numerator = np.zeros(W.shape[0])
        W_update_denominator = 0.0
        for i in range(len(V)):
            W_update_numerator = W_update_numerator + B[i][:, j] - ((W + V[i]) @ A[i])[:, j]
            W_update_denominator += A[i][j, j]
        temp = W[:, j] + W_update_numerator / W_update_denominator
        temp[temp < 1e-16] = 1e-16
        W[:, j] = temp
    return W


#@jit(nopython=True)
def _update_V_HALS(A, B, W, V, value_lambda):

    for j in range(V[0].shape[1]):
        for i in range(len(V)):
            temp = V[i][:, j] + (B[i][:, j] - (W + (1 + value_lambda) * V[i]) @ A[i][:, j]) / ((1 + value_lambda) * A[i][j, j])
            temp[temp < 1e-16] = 1e-16
            V[i][:, j] = temp

    return V
"""
def nnlsm_blockpivot(A, B, is_input_prod=False, init=None):
    """ Nonnegativity-constrained least squares with block principal pivoting method and column grouping
    Solves min ||AX-B||_2^2 s.t. X >= 0 element-wise.
    J. Kim and H. Park, Fast nonnegative matrix factorization: An active-set-like method and comparisons,
    SIAM Journal on Scientific Computing,
    vol. 33, no. 6, pp. 3261-3281, 2011.
    Parameters
    ----------
    A : numpy.array, shape (m,n)
    B : numpy.array or scipy.sparse matrix, shape (m,k)
    Optional Parameters
    -------------------
    is_input_prod : True/False. -  If True, the A and B arguments are interpreted as
            AtA and AtB, respectively. Default is False.
    init: numpy.array, shape (n,k). - If provided, init is used as an initial value for the algorithm.
            Default is None.
    Returns
    -------
    X, (success, Y, num_cholesky, num_eq, num_backup)
    X : numpy.array, shape (n,k) - solution
    success : True/False - True if the solution is found. False if the algorithm did not terminate
            due to numerical errors.
    Y : numpy.array, shape (n,k) - Y = A.T * A * X - A.T * B
    num_cholesky : int - the number of Cholesky factorizations needed
    num_eq : int - the number of linear systems of equations needed to be solved
    num_backup: int - the number of appearances of the back-up rule. See SISC paper for details.
    """
    if is_input_prod:
        AtA = A
        AtB = B
    else:
        AtA = A.T.dot(A)
        if sps.issparse(B):
            AtB = B.T.dot(A)
            AtB = AtB.T
        else:
            AtB = A.T.dot(B)

    (n, k) = AtB.shape
    MAX_ITER = n * 5

    if init is not None:
        PassSet = init > 0
        X, num_cholesky, num_eq = normal_eq_comb(AtA, AtB, PassSet)
        Y = AtA.dot(X) - AtB
    else:
        X = np.zeros([n, k])
        Y = -AtB
        PassSet = np.zeros([n, k], dtype=bool)
        num_cholesky = 0
        num_eq = 0

    p_bar = 3
    p_vec = np.zeros([k])
    p_vec[:] = p_bar
    ninf_vec = np.zeros([k])
    ninf_vec[:] = n + 1
    not_opt_set = np.logical_and(Y < 0, ~PassSet)
    infea_set = np.logical_and(X < 0, PassSet)

    not_good = np.sum(not_opt_set, axis=0) + np.sum(infea_set, axis=0)
    not_opt_colset = not_good > 0
    not_opt_cols = not_opt_colset.nonzero()[0]

    big_iter = 0
    num_backup = 0
    success = True
    while not_opt_cols.size > 0:
        big_iter += 1
        if MAX_ITER > 0 and big_iter > MAX_ITER:
            success = False
            break

        cols_set1 = np.logical_and(not_opt_colset, not_good < ninf_vec)
        temp1 = np.logical_and(not_opt_colset, not_good >= ninf_vec)
        temp2 = p_vec >= 1
        cols_set2 = np.logical_and(temp1, temp2)
        cols_set3 = np.logical_and(temp1, ~temp2)

        cols1 = cols_set1.nonzero()[0]
        cols2 = cols_set2.nonzero()[0]
        cols3 = cols_set3.nonzero()[0]

        if cols1.size > 0:
            p_vec[cols1] = p_bar
            ninf_vec[cols1] = not_good[cols1]
            true_set = np.logical_and(not_opt_set, np.tile(cols_set1, (n, 1)))
            false_set = np.logical_and(infea_set, np.tile(cols_set1, (n, 1)))
            PassSet[true_set] = True
            PassSet[false_set] = False
        if cols2.size > 0:
            p_vec[cols2] = p_vec[cols2] - 1
            temp_tile = np.tile(cols_set2, (n, 1))
            true_set = np.logical_and(not_opt_set, temp_tile)
            false_set = np.logical_and(infea_set, temp_tile)
            PassSet[true_set] = True
            PassSet[false_set] = False
        if cols3.size > 0:
            for col in cols3:
                candi_set = np.logical_or(
                    not_opt_set[:, col], infea_set[:, col])
                to_change = np.max(candi_set.nonzero()[0])
                PassSet[to_change, col] = ~PassSet[to_change, col]
                num_backup += 1

        (X[:, not_opt_cols], temp_cholesky, temp_eq) = normal_eq_comb(
            AtA, AtB[:, not_opt_cols], PassSet[:, not_opt_cols])
        num_cholesky += temp_cholesky
        num_eq += temp_eq
        X[abs(X) < 1e-16] = 0
        Y[:, not_opt_cols] = AtA.dot(X[:, not_opt_cols]) - AtB[:, not_opt_cols]
        Y[abs(Y) < 1e-16] = 0

        not_opt_mask = np.tile(not_opt_colset, (n, 1))
        not_opt_set = np.logical_and(
            np.logical_and(not_opt_mask, Y < 0), ~PassSet)
        infea_set = np.logical_and(
            np.logical_and(not_opt_mask, X < 0), PassSet)
        not_good = np.sum(not_opt_set, axis=0) + np.sum(infea_set, axis=0)
        not_opt_colset = not_good > 0
        not_opt_cols = not_opt_colset.nonzero()[0]

    return X, (success, Y, num_cholesky, num_eq, num_backup)


def normal_eq_comb(AtA, AtB, PassSet=None):
    """ Solve many systems of linear equations using combinatorial grouping.
    M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450
    Parameters
    ----------
    AtA : numpy.array, shape (n,n)
    AtB : numpy.array, shape (n,k)
    Returns
    -------
    (Z,num_cholesky,num_eq)
    Z : numpy.array, shape (n,k) - solution
    num_cholesky : int - the number of unique cholesky decompositions done
    num_eq: int - the number of systems of linear equations solved
    """
    num_cholesky = 0
    num_eq = 0
    if AtB.size == 0:
        Z = np.zeros([])
    elif (PassSet is None) or np.all(PassSet):
        Z = nla.solve(AtA, AtB)
        num_cholesky = 1
        num_eq = AtB.shape[1]
    else:
        Z = np.zeros(AtB.shape)
        if PassSet.shape[1] == 1:
            if np.any(PassSet):
                cols = PassSet.nonzero()[0]
                Z[cols] = nla.solve(AtA[np.ix_(cols, cols)], AtB[cols])
                num_cholesky = 1
                num_eq = 1
        else:
            #
            # Both _column_group_loop() and _column_group_recursive() work well.
            # Based on preliminary testing,
            # _column_group_loop() is slightly faster for tiny k(<10), but
            # _column_group_recursive() is faster for large k's.
            #
            grps = _column_group_recursive(PassSet)
            for gr in grps:
                cols = PassSet[:, gr[0]].nonzero()[0]
                if cols.size > 0:
                    ix1 = np.ix_(cols, gr)
                    ix2 = np.ix_(cols, cols)
                    #
                    # scipy.linalg.cho_solve can be used instead of numpy.linalg.solve.
                    # For small n(<200), numpy.linalg.solve appears faster, whereas
                    # for large n(>500), scipy.linalg.cho_solve appears faster.
                    # Usage example of scipy.linalg.cho_solve:
                    #Z[ix1] = sla.cho_solve(sla.cho_factor(AtA[ix2]),AtB[ix1])
                    #
                    Z[ix1] = nla.solve(AtA[ix2], AtB[ix1])
                    num_cholesky += 1
                    num_eq += len(gr)
                    num_eq += len(gr)
    return Z, num_cholesky, num_eq


def _column_group_recursive(B):
    """ Given a binary matrix, find groups of the same columns
        with a recursive strategy
    Parameters
    ----------
    B : numpy.array, True/False in each element
    Returns
    -------
    A list of arrays - each array contain indices of columns that are the same.
    """
    initial = np.arange(0, B.shape[1])
    return [a for a in column_group_sub(B, 0, initial) if len(a) > 0]


def column_group_sub(B, i, cols):
    vec = B[i][cols]
    if len(cols) <= 1:
        return [cols]
    if i == (B.shape[0] - 1):
        col_trues = cols[vec.nonzero()[0]]
        col_falses = cols[(~vec).nonzero()[0]]
        return [col_trues, col_falses]
    else:
        col_trues = cols[vec.nonzero()[0]]
        col_falses = cols[(~vec).nonzero()[0]]
        after = column_group_sub(B, i + 1, col_trues)
        after.extend(column_group_sub(B, i + 1, col_falses))
    return after
