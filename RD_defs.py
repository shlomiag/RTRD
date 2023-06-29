"""
Common definitions for working with RD problems.
e.g., define an RD problem, how it should be computed, represent RD solutions (roots),
default parameter values, etc.

Noteworthy constructs:
    RD_problem              Represents an RD problem.
    RD_compute_params       Represents the parameters used to solve a problem numerically.
    RD_sol_point            A solutions point (root) of an RD problem. e.g., computed numerically.

Utility functions:
    load_solutions(), save_solutions()
                            Load and save computed solutions.
    log2_xrange_domain()    Generate a grid which is evenly-spaced in logarithmic scale.
    debug_print()           A customizable print function for debug info.
    KL()                    The Kullback-Leibler divergence.

Configuration values for modules in this package are provided below.
"""
import lzma
import os
import pickle
import glob
import numpy as np
import warnings
from os import getpid
from collections.abc import Iterable
from scipy.stats import entropy
from datetime import datetime

# Values below this are considered "almost zero".
DEFAULT_NEAR_ZERO_VALUE = 1.e-9

# A default stopping condition for Blahut-Arimoto's algorithm [2] (L_infty-distance between consecutive iterates).
DEFAULT_STOPPING_CONDITION_FOR_BA = 1.e-9

# BA is forced to stop if exceeds this many iterations.
MAX_BA_ITERATIONS_ALLOWED = 10 ** 6

# Computation methods implemented by this software package --- see `RD_solver.py`.
ADMISSIBLE_COMPUTATION_METHODS = ('reverse annealing', 'independent', 'diff-eq fixed order and step size')

# A default folder for storing solutions; used on save_solutions() and load_solutions().
DEFAULT_LOAD_SAVE_DIR = os.path.dirname(os.path.realpath(__file__))

# Emit debug info and enable debug on unexpected edge-cases; value is used across several package modules.
_DEBUG = True

_np_float_type = np.longdouble

# Attributes for debug_print() below.
PRINT_TO_STDOUT = True
PID_PREFIX = False
TIMESTAMP_PREFIX = False


def debug_print(msg: str, force_flush: bool = False, force_to_stdout: bool = False) -> None:
    """
    A debug print.

    Fine-control can be achieved with the global variables above.

    Parameters:
        Print to standard output if `PRINT_TO_STDOUT` or `force_to_stdout`.
        Prepend process ID if `PID_PREFIX` --- for multi-processing computation.
        Prepend a timestamp if `TIMESTAMP_PREFIX`.
        Flush stodout if `force_flush`.

    """
    if PID_PREFIX:
        msg_out = "PID %d:\t" % getpid() + str(msg)
    else:
        msg_out = str(msg)

    if TIMESTAMP_PREFIX:
        msg_out = str(datetime.now()).split('.')[0] + " " + msg_out

    if PRINT_TO_STDOUT or force_to_stdout:
        print(msg_out, flush=force_flush)


def KL(p1, p2):
    """
    The Kullback-Leibler divergence,    (in base 2)
                D[ p1 || p2 ] :=    sum p1(x) log p1(x)/p2(x)

    Multidimensional inputs are treated as *joint* probability distributions.
    """
    return entropy(p1.flatten(), p2.flatten(), base=2.)


def _is_probability_distribution(dist: np.ndarray, eps: float = 10 * DEFAULT_NEAR_ZERO_VALUE) -> bool:
    """ True if `dist` is a normalized probability distribution. """
    return _is_equal(np.sum(dist), 1, eps=eps) and (dist >= 0).all()


def _is_equal(x: np.ndarray, y: np.ndarray, eps: float = DEFAULT_NEAR_ZERO_VALUE) -> bool:
    """ Return True if `x` and `y` are equal up to an `eps` distance in L_infty, False otherwise. """
    return (np.absolute(x - y) < eps).all()


def log2_xrange_domain(log_start: float, log_stop: float, num: int = 50, inverse_order=False, basis=2.0) -> float:
    """
    Yield evenly spaced points in [log_start, log_stop) in logarithm scale.
    """
    float_log_interval = 1.0 * (log_stop - log_start)
    for i in range(num):
        if inverse_order:
            yield basis ** (log_start + float_log_interval * (num - 1 - i) / num)
        else:
            yield basis ** (log_start + float_log_interval * i / num)


def _measure_machines_precision() -> int:
    """
    Find the smallest significand i of a floating point representation,

        1 + 2**-i == 1
    """
    addend = np.array(.5)
    i = 1
    while np.array(1.) + addend != np.array(1.):
        addend /= 2
        i += 1

    return i


# Warn user if precision requirements are too stringent.
if abs(np.log2(DEFAULT_NEAR_ZERO_VALUE)) + 5 > _measure_machines_precision():
    warnings.warn("Possible precision issue: an accuracy of {} binary digits is assumed by code, but only {} are "
                  "supported by machine's float type.".format(int(abs(np.log2(DEFAULT_NEAR_ZERO_VALUE))),
                                                              _measure_machines_precision()))

if abs(np.log2(DEFAULT_STOPPING_CONDITION_FOR_BA)) + 5 > _measure_machines_precision():
    warnings.warn("Possible precision issue: a default accuracy of {} binary digits on BA's convergence, but only {} "
                  "are supported by machine's float type.".format(int(abs(np.log2(DEFAULT_STOPPING_CONDITION_FOR_BA))),
                                                                  _measure_machines_precision()))

__COMPUTE_PARAMS = (('computation_method', "The method used to solve the problem numerically; "
                                           "must be in {}.".format(ADMISSIBLE_COMPUTATION_METHODS)),
                    ('BA_stopping_condition', "Stopping condition for Blahut-Arimoto's algorithm (supremum norm)."),
                    ('uniform_initial_conditions', "If True then BA is initialized uniformly, otherwise randomly."),
                    ('log2_beta_range', "[low, hi], such that a solution is computed for beta in [2**low, 2**hi]."),
                    ('sample_num', "Number of points in grid when computing with BA (use `step_size` for RTRD)."),
                    ('max_BA_iters_allowed', "Maximal number of BA iterations allowed."),
                    ('order', "Order of Taylor method used in RD Root-Tracking."),
                    ('step_size', "Size size of RD Root-Tracking."),
                    ('cluster_mass_threshold', "Cluster mass threshold for RD Root-Tracking."))

_COMPUTE_PARAM_PROPS = list(k for k, v in __COMPUTE_PARAMS)

_DEFAULT_PARAM_VALS = dict((('BA_stopping_condition', DEFAULT_STOPPING_CONDITION_FOR_BA),
                            ('max_BA_iters_allowed', MAX_BA_ITERATIONS_ALLOWED),
                            ('uniform_initial_conditions', True)))

for k in _DEFAULT_PARAM_VALS.keys():
    assert k in _COMPUTE_PARAM_PROPS


class RD_compute_params(object):
    """ This docstring is generated below. """
    def __init__(self, computation_method, **kwargs):
        assert computation_method in ADMISSIBLE_COMPUTATION_METHODS, \
            "Unexpected computation method '{}' at RD_compute_params constructor. " \
            "Should be in {}".format(computation_method, ADMISSIBLE_COMPUTATION_METHODS)
        self.computation_method = computation_method

        log2_beta_range = kwargs.get('log2_beta_range')
        if log2_beta_range is not None:
            assert log2_beta_range[0] < log2_beta_range[1], "Unexpected beta range: {}".format(log2_beta_range)

        sample_num = kwargs.get('sample_num')
        assert sample_num is None or (0 < sample_num == int(sample_num))

        # Assign optional arguments. First item in list is `computation_method`
        assert 'computation_method' == _COMPUTE_PARAM_PROPS[0]
        for k in _COMPUTE_PARAM_PROPS[1:]:
            setattr(self, k, kwargs.get(k, _DEFAULT_PARAM_VALS.get(k)))

    def __repr__(self) -> str:
        stringify = lambda p: "'" + p + "'" if isinstance(p, str) else str(p)
        prop_as_str = ', '.join(prop + "=" + stringify(getattr(self, prop, None)) for prop in _COMPUTE_PARAM_PROPS
                                if getattr(self, prop) is not None)
        return "RD_compute_params(" + prop_as_str + ")"

    @property
    def computation_method(self) -> str:
        return self._computation_method

    @computation_method.setter
    def computation_method(self, val: str) -> None:
        assert val in ADMISSIBLE_COMPUTATION_METHODS, "Invalid computation method: '{}'".format(val)
        self._computation_method = val


RD_compute_params.__doc__ = \
    """
    Parameters used to compute the solutions of an RD problem.
    
    Arguments:\n\t{}{}
    
    Optional arguments:\n\t{}
    """.format(('`' + __COMPUTE_PARAMS[0][0] + '`').ljust(30), __COMPUTE_PARAMS[0][1],
               '\n\t'.join(list('{}{}'.format(('`' + k + '`').ljust(30), v) for k, v in __COMPUTE_PARAMS[1::])))


# Some default values for computation parameters, using reverse deterministic annealing.
default_compute_params = RD_compute_params(BA_stopping_condition=1.e-7, max_BA_iters_allowed=MAX_BA_ITERATIONS_ALLOWED,
                                           uniform_initial_conditions=True, computation_method='reverse annealing',
                                           log2_beta_range=[-1, 8], sample_num=5000)


def _make_column_vec(v: np.ndarray) -> np.ndarray:
    """ A representation of `v` as a column vector. """
    v_col = v.reshape((-1, 1))
    assert len(v_col.shape) == 2 and v_col.shape[1] == 1
    return v_col


class RD_problem(object):
    """
    Represents an RD problem definition.

    An RD problem is defined by:

        D               A distortion matrix, indexed (t, x).
        p_x             The probability distribution of the source X, represented as a column vector.
        problem_name    The problem's name (optional).
    """
    def __init__(self, p_x: np.ndarray, D: np.ndarray, problem_name: str = ""):
        """ See class documentation for details. """
        assert D.shape[1] == len(p_x), \
            "Shape {} of distortion matrix is incompatible with length {} of X-marginal.".format(D.shape, len(p_x))
        assert _is_probability_distribution(p_x), "Expecting a normalized probability distribution, got: {}".format(p_x)
        self.p_x, self.D, self.problem_name = _make_column_vec(p_x), D, problem_name

    def __repr__(self) -> str:
        return "RD_problem(p_x={}, D={})".format( repr(self.p_x), repr(self.D)) if self.problem_name == "" \
            else "RD_problem(problem_name='{}', p_x={}, D={})".format(self.problem_name, repr(self.p_x), repr(self.D))

    @property
    def X_dim(self) -> int:
        """ Source alphabet size. """
        return len(self.p_x)

    @property
    def T_dim(self) -> int:
        """ Reproduction alphabet size. """
        return self.D.shape[0]


class RD_sol_point(object):
    """
    Represents a particular solution point of an RD problem.

    A solution is defined by:
        p_t         A cluster marginal, represented as a column vector.
        beta        at which the solution was computed.
        iter_count  Number of BA iterations used to obtain solution (optional).
    """
    def __init__(self, p_t: np.ndarray, beta: float, iter_count=None):
        # Do *not* enforce normalization, since cluster-marginals p_t are not
        # necessarily normalized when computed with RTRD.
        self.p_t, self.beta, self.iter_count = p_t.reshape((len(p_t), 1)), beta, iter_count

    def __repr__(self) -> str:
        return "RD_sol_point(p_t={}, beta={})".format(self.p_t, self.beta) if self.iter_count is None \
            else "RD_sol_point(p_t={}, beta={}, iter_count={})".format(self.p_t, self.beta, self.iter_count)

    def _Z(self, prob: RD_problem) -> np.ndarray:
        """ Partition function. """
        exp_minus_beta_d = np.exp(-self.beta * prob.D)
        return np.sum(self.p_t * exp_minus_beta_d, axis=0, keepdims=True)

    def _sim_to_BA_Jacobian_matrix_wrt_direct_enc(self, prob: RD_problem) -> np.ndarray:
        """
        A matrix similar to the Jacobian matrix of the BA operator in *direct-encoder*
        coordinates, evaluated at self. This is [1, Eq. (6.3)] (at Proposition 23).

        Indexing: T_dim blocks of size X_dim along each axis.

        Note:   When p(x) vanishes nowhere, the Jacobian can be obtained from
                this matrix by conjugating with p(x).
        """
        X_dim, T_dim = prob.X_dim, prob.T_dim
        size = X_dim * T_dim
        block_diagonal = np.kron(np.identity(T_dim), np.ones(shape=(X_dim, X_dim)))

        # Indexed (t, x):
        direct_enc = self.direct_encoder(prob=prob)
        # flatten is row-major by default.
        tiled_direct_enc = np.tile(np.expand_dims(direct_enc.flatten(), 0).T, reps=(1, size))

        # Indexed (x, t):
        inv_enc = self.inverse_encoder(prob=prob)       # already contains the marginal p_x.
        temp_array = np.zeros(shape=(X_dim, size))
        for t in range(T_dim):
            temp_array[:, (t * X_dim):((t + 1) * X_dim)] = inv_enc[0:X_dim, t].reshape((X_dim, 1))
        inv_enc_smeared = np.tile(temp_array, reps=(T_dim, 1))

        return (block_diagonal - tiled_direct_enc) * inv_enc_smeared

    def BA_Jacobian_eigenvals_wrt_direct_enc(self, prob: RD_problem) -> np.ndarray:
        """ Eigenvalues of the Jacobian [1, Eq. (6.3)] of the BA operator in *encoder* coordinates. """
        assert not (prob.p_x == 0).any(), "Calculation of Jacobian's eigenvalues with respect to these coordinates " \
                                          "implemented only for problems with non-vanishing X-marginal."

        eigs = np.linalg.eigvals(self._sim_to_BA_Jacobian_matrix_wrt_direct_enc(prob))

        if _DEBUG and (np.abs(np.imag(eigs)) > 1e-14).any():
            print("Unexpected imaginary part of eigenvalues:\n", eigs)
            import pdb;
            pdb.set_trace()

        return np.real(eigs)

    def BA_Jacobian_matrix_wrt_marginal(self, prob: RD_problem) -> np.ndarray:
        """
        The Jacobian matrix [1, Eq. (5.7)] of the BA operator in cluster-marginal
        coordinates, evaluated at self.
        """
        X_dim, T_dim, D = prob.X_dim, prob.T_dim, prob.D
        Z = self._Z(prob).reshape((1, X_dim, 1))
        p_x = prob.p_x.reshape((1, X_dim, 1))
        d1 = np.tile(np.expand_dims(D, 2), (1, 1, T_dim))
        d2 = np.tile(np.expand_dims(D.T, 0), (T_dim, 1, 1))
        return np.identity(n=T_dim) - \
               self.p_t * np.sum(p_x / Z ** 2 * np.exp(-self.beta * (d1 + d2)), axis=1, keepdims=False)

    def BA_Jacobian_eigenvals_wrt_marginal(self, prob: RD_problem) -> np.ndarray:
        """ Eigenvalues of The Jacobian matrix [1, Eq. (5.7)] of the BA operator in marginal coordinates. """
        return np.linalg.eigvals(self.BA_Jacobian_matrix_wrt_marginal(prob))

    def direct_encoder(self, prob: RD_problem) -> np.ndarray:
        """ Test-channel, aka encoder or direct-encoder. Indexed (t, x). """
        exp_minus_beta_d = np.exp(-self.beta * prob.D)
        return self.p_t * exp_minus_beta_d / np.sum(self.p_t * exp_minus_beta_d, axis=0, keepdims=True)

    def inverse_encoder(self, prob: RD_problem) -> np.ndarray:
        """
        The inverse-encoder p(x|t) at self; indexed (x, t).
        """
        if (self.p_t < DEFAULT_NEAR_ZERO_VALUE).any():
            warnings.warn("Inverse encoder need not be normalized for clusters of small mass.")

        exp_minus_beta_d = np.exp(-self.beta * prob.D)  # indexed (t, x)
        p_x = prob.p_x.reshape((1, prob.X_dim))
        return (p_x * exp_minus_beta_d / np.sum(self.p_t * exp_minus_beta_d, axis=0, keepdims=True)).T

    def joint_TX(self, prob: RD_problem) -> np.ndarray:
        """ The joint probability distribution matrix of T and X; indexed (t, x). """
        # p(x, t) = p(t|x) p(x).    Direct encoder p(t|x) indexed (t, x).
        p_x = prob.p_x.reshape((1, prob.X_dim))
        return self.direct_encoder(prob=prob) * p_x

    def distortion(self, prob: RD_problem) -> np.ndarray:
        """ Expected distortion D at self. """
        return np.sum(self.joint_TX(prob=prob) * prob.D)

    def rate(self, prob: RD_problem) -> float:
        """ Expected rate (bits per sample) at self. """
        # Marginals' product indexed (t, x).
        p_x = prob.p_x.reshape((1, prob.X_dim))
        marginals_product = p_x * self.p_t.reshape((prob.T_dim, 1))

        return KL(self.joint_TX(prob=prob), marginals_product)


# Short names used for auto-generating filenames, below.
__SHORT_NAMES = dict((('reverse annealing', 'rev-ann'),
                      ('independent', 'ind'),
                      ('diff-eq fixed order and step size', 'vanilla_Taylor_method')))

for n in ADMISSIBLE_COMPUTATION_METHODS:
    assert n in __SHORT_NAMES, "{} not provided a short-name.".format(n)


def _generate_filename(prob, params, add_timestamp, output_dir):
    """ Auto-generate filename (without extension) based on problem & its parameters. """
    if hasattr(prob, 'filename'):
        # Already have a filename.
        normalized_prob_name = getattr(prob, 'filename').rstrip('.pkl.lzma')
    else:
        # No filename, generate a new one.
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S ') if add_timestamp else ""
        normalized_prob_name = timestamp + prob.problem_name.replace('\\', '').replace('$', '')
        normalized_prob_name += ", prec={:.3g}, {}".format(params.BA_stopping_condition,
                                                           __SHORT_NAMES[params.computation_method])

    return os.path.join(output_dir, normalized_prob_name)


def _load_all(in_file):
    """ Yield one pickled object at a time from in_file. """
    while True:
        try:
            yield pickle.load(in_file)
        except Exception as e:
            if isinstance(e, EOFError):
                break
            if isinstance(e, UnicodeDecodeError):
                print("Re-attempting to load object in utf8 encoding.")
                yield pickle.load(in_file, encoding="utf8")
            else:
                raise e


def load_solutions(filename: str = None, verbose: bool = True):
    """
    Load RD solutions and definitions from file.

    `filename` is optional; loads from most recent solutions file under DEFAULT_LOAD_SAVE_DIR if None.

    Usage:
        prob, params, sols = load_solutions()
    """
    if filename is None:
        list_of_files = glob.glob(os.path.join(DEFAULT_LOAD_SAVE_DIR, '*.lzma'))
        latest_file = max(list_of_files, key=os.path.getctime)
        filename = os.path.join(DEFAULT_LOAD_SAVE_DIR, latest_file)

    with lzma.LZMAFile(filename, 'rb') as in_file:
        items = _load_all(in_file)
        prob, params = next(items), next(items)
        sols = sorted([s for s in items], key=lambda s: s.beta)

    # filename attribute of prob used when saving plots.
    prob.filename = os.path.split(filename)[-1]

    if verbose:
        print('Loaded {} solution points from "{}"'.format(len(sols), prob.filename))

    return prob, params, sols


def save_solutions(prob: RD_problem, params: RD_compute_params, sols: [RD_compute_params],
                   verbose: bool = True, **kwargs) -> None:
    """
    Saves solutions.

    Optional arguments:
        filename                File name is auto-generated if not provided.
        output_dir              Output directory for auto-generated file names; defaults to DEFAULT_LOAD_SAVE_DIR.
        add_timestamp           Prepend a timestamp to auto-generated file names.
        overwrite_existing      Overwrite existing output file.
        verbose                 Acknowledge when done saving.

    Usage:
        save_solutions(prob, params, sols)
    """
    assert isinstance(prob, RD_problem), "First argument expected to be an RD_problem instance, " \
                                         "but is {}.".format(type(prob))
    assert isinstance(params, RD_compute_params), "Second argument expected to be an RD_compute_params instance, " \
                                                  "but is {}.".format(type(params))
    assert isinstance(sols, Iterable), "Third argument expected to be an Iterable of RD-solutions, " \
                                       "but is {}.".format(type(sols))
    for i, s in enumerate(sols):
        assert isinstance(s, RD_sol_point), \
            "sols[{}] expected to be an RD_sol_point instance, but is {}.".format(i, type(s))

    filename = kwargs.get('filename', None)
    if filename is None:
        output_dir = kwargs.get('output_dir', DEFAULT_LOAD_SAVE_DIR)
        add_timestamp = kwargs.get('add_timestamp', True)
        filename = _generate_filename(prob, params, add_timestamp=add_timestamp, output_dir=output_dir) + '.pkl.lzma'

    if os.path.exists(filename) and not kwargs.get('overwrite_existing'):
        print("Filename {} already exists! Please specify 'overwrite_existing=True' to overwrite.")
        return

    with lzma.LZMAFile(filename, 'wb') as out_file:
        pickle.dump(prob, out_file)
        pickle.dump(params, out_file)
        # Saving solutions one-by-one for compatibility with heavy computation scenarios.
        for s in sols:
            pickle.dump(s, out_file)

    if verbose:
        print("{} solutions written successfully to file: {}".format(len(sols), filename))
