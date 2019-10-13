"""
This module contains ops to compute quantities related to matrices.
"""
import numpy as np
import torch as th
from tqdm import tqdm

from .. import backend as A
from ..utils import glog as log
from ..utils.decorators import coroutine
from ..core.events import DoneEvent


def lanczos_tridiagonalization_slow(H, q=None, K=None):
    """
    Given a symmetric matrix, factorize it as

    .. math::

        AQ = QT

    where :math:`Q` is orthogonal, and :math:`T` is tridiagonal. q would be the
    first column of Q. It is possible that rank(Q) is less than rank(H), since
    the algorithm may terminate when an invariant subspace is found. This could
    be an intermediate step to obtain the eigenvalues of H, since T is a matrix
    that is similar with H; it is also used (as an intermediate step) to
    approximate the eigenspectrum of H, when H is too large to compute its
    eigenspectrum exactly.

    More in *The Full Spectrum of Deep Net Hessians At Scale: Dynamics with
    Sample Size*, https://arxiv.org/abs/1811.07062.

    Args:
        H: a symmetric N X N matrix
        q: a N X 1 vector. If None, one will be created randomly.
        K: an integer. The number of iterations to run. If None, would run until
           an invariant subspace is found, or N iterations.

    Returns:
        eig: a tuple of two tensor: a K dimension vector contains eigenvalues
             of H; a K X N matrix contains eigenvectors.
    """
    N = A.get_shape(H)[0]

    if K is None:
        K = N

    if K > 10 ** 4:
        log.warning("Iteration number K {} may be too large".format(K))


    T = A.zeros((K, K))
    alpha = A.zeros(K)
    beta = A.zeros(K)
    V = A.zeros((N, K))

    k = 0
    beta_k = 1
    while beta_k != 0 and k < K:
        if k == 0:
            if q is None:
                # Sample a vector from N(0, I)
                v = A.randn((N,))
                v = v / A.nn.l2_norm(v)
                w = H @ v
            else:
                v = q
        else:
            w = H @ v - beta[k-1] * V[:, k-1]

        alpha_k = v @ w
        w = w - alpha_k * v
        w = w - V @ V.t() @ w
        beta_k = A.nn.l2_norm(w)

        V[:, k] = v
        alpha[k] = alpha_k
        beta[k] = beta_k

        v = w / beta_k

        k += 1

    # Construct T and compute its eigenvalues.
    diag_idxs = A.range(0, K)
    T[diag_idxs, diag_idxs] = alpha
    lower_diag_idxs = (diag_idxs + 1)[:-1]
    T[lower_diag_idxs, diag_idxs[:-1]] = beta[:-1]
    upper_diag_idxs = (diag_idxs - 1)[1:]
    T[upper_diag_idxs, diag_idxs[1:]] = beta[:-1]

    return A.symeig(T)


@coroutine
def lanczos_tridiagonalization_fast_cr(N, q=None, K=None, return_beta_k=False):
    """
    Coroutine version of the Lanczos tridiagonalization algorithm that
    delegates Hessian vector product. See `lanczos_tridiagonalization_fast`.
    """
    if K is None:
        K = N

    if K > 10 ** 4:
        log.warning("Iteration number K {} may be too large".format(K))


    T = A.zeros((K, K))
    alpha = A.zeros(K)
    beta = A.zeros(K)

    k = 0
    beta_k = 1

    pbar = tqdm(total=K)
    while beta_k != 0 and k < K:
        if k == 0:
            if q is None:
                # Sample a vector from N(0, I)
                v = A.randn((N,))
                # v = A.cast(v, th.float64) # TODO: fix here. A stub for adding
                # double precision support in the future.
                v = v / A.nn.l2_norm(v)
            else:
                v = q

            w = yield v # Delegate the computation of Hessian vector products up.
            # w = hessian_vector_product(H, v)
        else:
            Hv = yield v
            w = Hv - beta[k-1] * v_prev
            # w = hessian_vector_product(H, v) - beta[k-1] * v_prev

        alpha_k = v @ w
        w = w - alpha_k * v
        beta_k = A.nn.l2_norm(w)

        alpha[k] = alpha_k
        beta[k] = beta_k

        v_prev = v
        v = w / beta_k

        k += 1
        pbar.update(1)

    pbar.close()

    # Construct T and compute its eigenvalues.
    diag_idxs = A.range(0, K)
    T[diag_idxs, diag_idxs] = alpha
    lower_diag_idxs = (diag_idxs + 1)[:-1]
    T[lower_diag_idxs, diag_idxs[:-1]] = beta[:-1]
    upper_diag_idxs = (diag_idxs - 1)[1:]
    T[upper_diag_idxs, diag_idxs[1:]] = beta[:-1]

    eigenvalues, eigenvectors = A.symeig(T)

    if return_beta_k:
        yield eigenvalues, eigenvectors, beta[-1]
    else:
        yield eigenvalues, eigenvectors


def lanczos_tridiagonalization_fast(H, q=None, K=None, return_beta_k=False):
    """
    Lanczos algorithm that does not re-orthogonalize in each iteration. The
    time and space complexity is significantly smaller.

    The last off diagonal entry can be used to provide error bound on the
    eigenvalue estimation. In that, if an error estimation is needed, specify
    `return_beta_k` as True.

    See also `lanczos_tridiagonalization_slow`.
    """
    N = A.get_shape(H)[0]

    if K is None:
        K = N

    if K > 10 ** 4:
        log.warning("Iteration number K {} may be too large".format(K))


    T = A.zeros((K, K))
    alpha = A.zeros(K)
    beta = A.zeros(K)

    k = 0
    beta_k = 1
    while beta_k != 0 and k < K:
        if k == 0:
            if q is None:
                # Sample a vector from N(0, I)
                v = A.randn((N,))
                v = v / A.nn.l2_norm(v)
            else:
                v = q

            w = H @ v
        else:
            w = H @ v - beta[k-1] * v_prev

        alpha_k = v @ w
        w = w - alpha_k * v
        beta_k = A.nn.l2_norm(w)

        alpha[k] = alpha_k
        beta[k] = beta_k

        v_prev = v
        v = w / beta_k

        k += 1

    # Construct T and compute its eigenvalues.
    diag_idxs = A.range(0, K)
    T[diag_idxs, diag_idxs] = alpha
    lower_diag_idxs = (diag_idxs + 1)[:-1]
    T[lower_diag_idxs, diag_idxs[:-1]] = beta[:-1]
    upper_diag_idxs = (diag_idxs - 1)[1:]
    T[upper_diag_idxs, diag_idxs[1:]] = beta[:-1]

    eigenvalues, eigenvectors = A.symeig(T)

    if return_beta_k:
        return eigenvalues, eigenvectors, beta[-1]
    else:
        return eigenvalues, eigenvectors


@coroutine
def lanczos_spectrum_approx_cr(N, iter_num, K, n_vec):
    """
    The version of Lanczos spectrum approximation that delegates the computation
    of Hessian vector product. For the algorithm, see
    `lanczos_spectrum_approx`. Notably, in the coroutine version, the Hessian
    passed in the normal version is `N`, the dimension of the Hessian instead.
    """
    eigenvalues = A.zeros((n_vec, iter_num))
    eigenvectors = A.zeros((n_vec, iter_num, iter_num))
    print("Running Lanczos algorithm ... ")
    for i in tqdm(range(n_vec)):
        f, ret = lanczos_tridiagonalization_fast_cr(N, K=iter_num)
        Hv = yield ret
        while True:
            v = f.send(Hv)
            if type(v) is tuple:
                # We have reached the end of the iteration
                f.close()
                break
            Hv = yield v
        # The last returned values from coroutine below are the eigenvalues and
        # eigenvectors computed.
        eigenvalues[i], eigenvectors[i] = v

        # Essentially, the above code computes the following line in the normal version.
        # eigenvalues[i], eigenvectors[i] = lanczos_tridiagonalization_fast(H, K=iter_num)
        # eigenvalues[i], eigenvectors[i] = lanczos_tridiagonalization_slow(H, K=iter_num)

    t = A.linspace(-1, 1, K)
    psi = A.zeros(K)
    sigma = 2 / ((iter_num - 1) * ((8 * np.log(1.25)) ** (1/2)))
    gaussian = lambda t: 1 / (sigma * (2 * np.pi) ** (1/2)) * np.e ** (-(1/2) * (t / sigma) ** 2)
    print("Running eigenspectrum approximation ... ")
    tol = 1e-08
    width = sigma * np.sqrt(-2.0 * np.log(tol))
    for k in tqdm(range(K)):
        mass = 0
        for i in range(n_vec):
            for j in range(iter_num):
                if abs(t[k] - eigenvalues[i, j]) > width:
                    continue
                for v in eigenvectors[i, :, j]:
                    if abs(v) > 10e-7:
                        break
                mass += v ** 2 * gaussian(t[k] - eigenvalues[i, j])
        mass /= n_vec
        psi[k] = mass


    # return psi
    yield psi, DoneEvent("Lanczos Spectrum Approximation Done.")


def lanczos_spectrum_approx(H, iter_num, K, n_vec):
    """
    The algorithm that uses Lanczos algorithm to approximate the eigenspectrum
    of a symmetric matrix H. For more details, refer to *The Full Spectrum of
    Deep Net Hessians At Scale: Dynamics with Sample Size*,
    https://arxiv.org/abs/1811.07062.

    Args:
        H: a symmetric matrix of which the eigenspectrum is to be approximated,
           and is in the range [-1, 1].
        iter_num: the number of iteration to run in the Lanczos algorithm.
        K: the number of eigenvalues used as nodes (sampled points), around
           which the density of eigenvalues is approximated.
        n_vec: the number of unit-norm vectors to randomly sample to serve as
        the starting point of the Lanczos algorithm.

    Return:
        psi: a vector of dimension K. Density of the spectrum of H evaluated at
             K evenly distributed points in the range [-1, 1].
    """
    shape = A.get_shape(H)
    N = shape[0]
    eigenvalues = A.zeros((n_vec, iter_num))
    eigenvectors = A.zeros((n_vec, iter_num, iter_num))
    for i in range(n_vec):
        eigenvalues[i], eigenvectors[i] = lanczos_tridiagonalization_fast(H, K=iter_num)
        # eigenvalues[i], eigenvectors[i] = lanczos_tridiagonalization_slow(H, K=iter_num)

    t = A.linspace(-1, 1, K)
    psi = A.zeros(K)
    sigma = 2 / ((iter_num - 1) * ((8 * np.log(1.25)) ** (1/2)))
    gaussian = lambda t: 1 / (sigma * (2 * np.pi) ** (1/2)) * np.e ** (-(1/2) * (t / sigma) ** 2)
    for k in range(K):
        mass = 0
        for i in range(n_vec):
            for j in range(iter_num):
                for v in eigenvectors[i, :, j]:
                    if abs(v) > 10e-7:
                        break
                mass += v ** 2 * gaussian(t[k] - eigenvalues[i, j])
        mass /= n_vec
        psi[k] = mass

    return psi


@coroutine
def center_unit_eig_normalization_cr(N, iter_num, kappa):
    """
    Given a symmetric matrix, normalize the eigenspectrum of it to [-1, 1].

    Args:
        N: int
            The number of parameters.
        iter_num: int
            The number of interation to run in the underlyiong Lanzcos
            algorithm.
        kappa: float
            Margin percentage.

    Return:
        Parameters to denormalize the Hessian computed from the grad.
    """
    f, ret = lanczos_tridiagonalization_fast_cr(N, K=iter_num, return_beta_k=True)
    Hv = yield ret
    while True:
        v = f.send(Hv)
        if type(v) is tuple:
            # We have reached the end of the iteration
            f.close()
            break
        Hv = yield v
    # The last returned values from coroutine below are the eigenvalues and
    # eigenvectors computed.
    eigenvalues, eigenvectors, beta_k = v

    # eigenvalues, eigenvectors, beta_k = lanczos_tridiagonalization_fast(H, K=M, return_beta_k=True)
    # The following lines incorporate error bounds through Ritz
    # approximation. For the derivation, refer to p. 575, _Matrix Computation_,
    # also p. 318, _The Symmetric Eigenvalue Problem_.
    eig_min = eigenvalues[0] - abs(beta_k * eigenvectors[-1][0])
    eig_max = eigenvalues[-1] + abs(beta_k * eigenvectors[-1][-1])
    delta = kappa * (eig_max - eig_min)
    eig_min = eig_min - delta
    eig_max = eig_max + delta
    c = (eig_min + eig_max) / 2
    d = (eig_max - eig_min) / 2

    # We only return the parameters c, d computed, since grad would be
    # recomputed normally afterwards. The normalization should happen there.
    yield c, d


def center_unit_eig_normalization(H, M, kappa):
    """
    Given a symmetric matrix, normalize the eigenspectrum of it to [-1, 1].

    Args:
        M: int
            The number of interation to run in the underlyiong Lanzcos
            algorithm.
        kappa: float
            Margin percentage.

    Return:
        Normalized matrix, and parameters to denormalize it
    """
    N = A.get_shape(H)[0]
    eigenvalues, eigenvectors, beta_k = lanczos_tridiagonalization_fast(H, K=M, return_beta_k=True)
    # The following lines incorporate error bounds through Ritz
    # approximation. For the derivation, refer to p. 575, _Matrix Computation_,
    # also p. 318, _The Symmetric Eigenvalue Problem_.
    eig_min = eigenvalues[0] - abs(beta_k * eigenvectors[-1][0])
    eig_max = eigenvalues[-1] + abs(beta_k * eigenvectors[-1][-1])
    delta = kappa * (eig_max - eig_min)
    eig_min = eig_min - delta
    eig_max = eig_max + delta
    c = (eig_min + eig_max) / 2
    d = (eig_max - eig_min) / 2
    return (H - c * A.eye(N)) / d, c, d


def center_unit_eig_denormalization(H, c, d):
    """
    Given a matrix, and two parameters (obtained when doing normalization using
    'center_unit_eig_normalization`), denormalize the eigenspectrum of matrix
    `H` to its original range.
    """
    N = A.get_shape(H)[0]
    return H * d + c * A.eye(N)
