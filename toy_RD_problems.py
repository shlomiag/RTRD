"""
Toy problems used throughout the paper, [1].

Toy problems plotted at [1]:
    Berger71_Example_2_7_3()        Defines Berger's [3] Example 2.7.3: Figure 2.7.6 and 2.7.6 there.
    ISIT2021_CSD_prob*              Defines the problem of [4, Figure 1].
    binary_source_with_Hamming_distortion*()
                                    Defines a binary source with a Hamming distortion measure. cf., [1, Section III.F].

Other noteworthy problems, from [3]:
    Berger71_Problem_2_8(), Berger71_Problem_2_9(), Berger71_Problem_2_10()

See also README.md, for usage.
"""
from RD_defs import *
from itertools import product
from copy import deepcopy


# The problem defined at the ISIT'21 CSD paper, [4, Figure 1].
_xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
_xhats = _xs / 2
# D is indexed (t, x).
_D = np.zeros(shape=(4, 4))
for _x, _t in product(range(4), range(4)):
    # D is indexed (t, x)
    _D[_t, _x] = np.sum((_xs[_x, :] - _xhats[_t, :]) ** 2)
ISIT2021_CSD_prob = RD_problem(problem_name='ISIT2021 CSD', p_x=np.array([0.4, 0.3, 0.2, 0.1]), D=_D / np.max(_D))
ISIT2021_CSD_compute_params = RD_compute_params(BA_stopping_condition=1.e-9, max_BA_iters_allowed=10 ** 6,
                                                uniform_initial_conditions=True, computation_method='reverse annealing',
                                                log2_beta_range=[-1, 6], sample_num=1000)
ISIT2021_CSD_compute_params_for_RTRD = \
    RD_compute_params(BA_stopping_condition=1.e-9, max_BA_iters_allowed=10 ** 6,
                      uniform_initial_conditions=True, computation_method='diff-eq fixed order and step size',
                      log2_beta_range=[-1, 5], step_size=.01, order=3, cluster_mass_threshold=.01)


def Berger71_Example_2_7_3(p: float, rho: float, computation_method: str = 'independent') -> \
        [RD_problem, RD_compute_params]:
    """
    Example 2.7.3 in Berger '71, [3]:

        A binary source with probability p, distortion with penalty rho for erasure.

    Figure 2.7.5 there is with (p, rho) = (0.5, 0.3), Figure 2.7.6 with (0.4, 0.3).
    The latter is reproduced by [1, Figure 6.2].

    For p = 1/2, this yields an *apparent* first-order transition between two
    solutions, at the edge of the RD-curve. For p < 1/2, it is at the middle of the
    RD-curve. cf., Theorem 2.4.2 there.
    """
    assert 0 <= p <= 1, "p={} expected to represent probability of Bernouli variable.".format(p)
    if not 0 < p < 0.5:
        print("Recommended values for p: between 0 to 1/2.")
    if not rho < p:
        print("Per Berger, the only interesting cases are those in which rho < p. Otherwise, the extra letter "
              "is of no use at any value of D, and the problem reduces to a simpler one.")
    assert rho >= 0, "Illegal value {} for distortion.".format(rho)

    p_x = np.array([p, 1 - p])
    D = np.array([[0, 1, rho], [1, 0, rho]]).transpose()
    prob_name = r"Berger '71, Example 2.7.3, $p={}, \rho={}$".format(p, rho)
    prob_name += ", Fig. 2.7.5" if p == 0.5 and rho == 0.3 else ", Fig. 2.7.6" if p == 0.4 and rho == 0.3 else ""
    prob = RD_problem(p_x=p_x, D=D, problem_name=prob_name)
    params = deepcopy(default_compute_params)
    params.computation_method = computation_method
    params.log2_beta_range = [-1, 5]
    return prob, params


# A few more interesting problems, from Berger [3]:

def Berger71_Problem_2_8(p: float = 0.1, computation_method: str = 'independent') -> \
        [RD_problem, RD_compute_params]:
    """
    Problem 2.8 in Berger '71, [3]:

    Berger:
        M = N = 3, optimal Q_1 vanishes only for intermediate D values.
    """
    assert 0 <= p <= 1, "Invalid value {}: p to represent probability.".format(p)
    if not (0 < p < 1. / 6.):
        print("Recommended values for p={} is 0 < p < 1/6.".format(p))

    p_x = np.array([(1 - p) / 2, p, (1 - p) / 2])
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    prob_name = r"Berger '71, Problem 2.8, $p={}$".format(p)
    prob = RD_problem(p_x=p_x, D=D, problem_name=prob_name)
    params = deepcopy(default_compute_params)
    params.computation_method = computation_method
    params.log2_beta_range = [-4, 5]
    return prob, params


def Berger71_Problem_2_9(p: float = 0.1, rho: float = 0.9, computation_method: str = 'independent') -> \
        [RD_problem, RD_compute_params]:
    """
    Problem 2.9 in Berger '71, [3]:

    Berger:
        M=3 letters, N=4 clusters: a cluster is non-zero only for intermediate D values.
    """
    assert 0 <= p <= 1, "Invalid value {}: p to represent probability.".format(p)
    if not (0 < p < 1. / 6.):
        print("Recommended values for p={} is 0 < p < 1/6.".format(p))
    if not rho < p:
        print("Recommended value is rho < p, got rho={}, p={} instead.".format(rho, p))
    assert rho >= 0, "Illegal value {} for distortion.".format(rho)

    p_x = np.array([(1 - p) / 2, p, (1 - p) / 2])
    D = np.array([[0, 1, 2, rho], [1, 0, 1, rho], [2, 1, 0, rho]]).transpose()
    prob_name = r"Berger '71, Problem 2.9, $p={}, \rho={}$".format(p, rho)
    prob = RD_problem(p_x=p_x, D=D, problem_name=prob_name)
    params = deepcopy(default_compute_params)
    params.computation_method = computation_method
    params.max_BA_iters_allowed = 10 ** 7
    params.log2_beta_range = [-6, 4]
    return prob, params


def Berger71_Problem_2_10(p: float = 0.1, epsilon1: float = 2.5e-3, epsilon2: float = 1.25e-3,
                          computation_method: str = 'independent') -> [RD_problem, RD_compute_params]:
    """
    Problem 2.10 in Berger '71, [3]:

        A cluster may depart from and return to the boundary more than once as D increases.
    """
    assert 0 <= p <= 1, "Invalid value {}: p to represent probability.".format(p)
    assert 0 <= epsilon1 and 0 <= epsilon2, \
        "Expecting non-negative values for distortion: epsilon1={}, epsilon2={}".format(epsilon1, epsilon2)
    if not (0 < p < 1. / 6.):
        print("Recommended values for p={} is 0 < p < 1/6.".format(p))
    if not (10 * epsilon1 < p and 10 * epsilon2 < p):
        print("Both epsilon1={} and epsilon2={} expected to be very small.")
    assert epsilon1 != epsilon2, "epsilon1={} and epsilon2={} expected to be distinct.".format(epsilon1, epsilon2)

    p_x = np.array([(1 - p) / 2, p / 2, p / 2, (1 - p) / 2])
    D = np.array([[0, 1, 1, 2], [1, 0, epsilon1, 1], [1, epsilon2, 0, 1], [2, 1, 1, 0]]).transpose()
    prob_name = r"Berger '71, Problem 2.10, $p={}, \epsilon_1={}, \epsilon_2={}$".format(p, epsilon1, epsilon2)
    prob = RD_problem(p_x=p_x, D=D, problem_name=prob_name)
    params = deepcopy(default_compute_params)
    params.computation_method = computation_method
    params.log2_beta_range = [-5, 10]
    return prob, params


def binary_source_with_Hamming_distortion(p: float) -> [RD_problem, RD_compute_params]:
    """ Problem definition and computation parameters for a binary source with a Hamming distortion measure. """
    assert 0. <= p < 0.5, "The parameter p < 0.5 is expected to represent the probability of a " \
                          "Bernoulli; got {}".format(p)
    prob = RD_problem(p_x=np.array([1 - p, p]), D=np.array([[0, 1], [1, 0]]),
                      problem_name=r"Binary source~Bernoulli({}) with a Hamming distortion".format(p))
    params = RD_compute_params(BA_stopping_condition=DEFAULT_STOPPING_CONDITION_FOR_BA,
                               computation_method='independent',
                               log2_beta_range=[-1, 5], uniform_initial_conditions=True, sample_num=3000)
    return prob, params


def binary_source_with_Hamming_distortion_for_RTRD(p: float) -> [RD_problem, RD_compute_params]:
    """ Problem definition and parameters for RTRD computation of a binary source with a Hamming distortion measure. """
    prob, params = binary_source_with_Hamming_distortion(p)
    params.computation_method = 'diff-eq fixed order and step size'
    params.cluster_mass_threshold = 0.01
    params.step_size = 0.01
    params.order = 3
    params.sample_num = None
    return prob, params

