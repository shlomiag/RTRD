"""
Specializes the computation of higher implicit multivariate derivatives [1, Algorithm 1] to RD.

Implements the formulas of [1, Theorem 4] for the higher derivative tensors of the RD operator,
$Id - BA_\beta$ [1, Eq. (1.2)], in marginal coordinates. This is provided via the
IdMinusBADerivativeTensorsInMarginalCoordinates class, to be used with the RootTracker
class from `root_tracking.py`.
"""
import sympy
import pickle
import lzma
import functools
import os
import numpy as np
from itertools import permutations
import root_tracking
import RD_defs


# Cached P_k polynomials are stored at this directory.
CACHE_BASEDIR = os.path.dirname(os.path.realpath(__file__))


def _derivation(expr, univariate_monomial_deriv):
    """
    The derivation of a sympy expression `expr`.

    On univariate monomials such as x**n it is defined by `univariate_monomial_deriv`.
    Numeric inputs should be first converted to a respective SymPy object, such as sympy.Integer instead of int.

    Adapted from        https://stackoverflow.com/a/58463715

    >>> x, y = sympy.symbols('x y')
    >>> D = sympy.Function('D')
    >>> _derivation(x + y, D)
    D(x) + D(y)
    """
    if expr.is_Add:
        # d(f+g) = d(f) + d(g)
        return sympy.Add(*[_derivation(current_arg, univariate_monomial_deriv) for current_arg in expr.args])

    if expr.is_Mul:
        # d(f*g) = d(f)*g + f*d(g)
        return sympy.Add(*[sympy.Mul(*(expr.args[:i] +
                                       (_derivation(current_arg, univariate_monomial_deriv), ) +
                                       expr.args[i + 1:])) for i, current_arg in enumerate(expr.args)])

    # Monomials x_i**j are handled at _dbar_deriv_for_univariate_monomial().
    return univariate_monomial_deriv(expr)


def _dbar_deriv_of_variable(variable):
    """
    Implements the derivation [1, Eq. (2.22)] on variables.

    Variables names expected to be a single letter followed by a non-negative int: 'x0', 'k1', 'y17', etc.
    """
    name = variable.name
    letter, num = name[0], int(name[1:])
    assert variable.is_Symbol and len(name) >= 2 and name[0].isalpha() and num >= 0, \
        "Unexpected variable: {}".format(variable.name)

    if num == 0:
        # The derivation of x0 vanishes, [1, Eq. (2.22)].
        return 0

    # The derivation of x_k, for k > 0, [1, Eq. (2.22)].
    x1 = sympy.symbols(letter + '1')
    x_next = sympy.symbols(letter + str(num + 1))
    return x1 * variable - x_next


def _dbar_deriv_for_univariate_monomial(univariate_monomial):
    """ The derivation [1, Eq. (2.22)] on univariate monomials, x_i**j."""
    if univariate_monomial.is_constant():
        return 0

    if not univariate_monomial.is_Pow:
        # A single variable. e.g., 'x0' or 'x17'.
        return _dbar_deriv_of_variable(univariate_monomial)

    # Must be of power > 1.
    assert len(univariate_monomial.args) == 2
    variable, exponent = univariate_monomial.args
    return exponent * variable**(exponent - 1) * _dbar_deriv_of_variable(variable)


def dbar_deriv(expr):
    """ The derivation [1, Eq. (2.22)] used to define the P_k polynomials of [1, Eq. (2.23)]. """
    return _derivation(sympy.expand(expr), _dbar_deriv_for_univariate_monomial)


def _cache_list_to_file(file_name: str, compress_data=True, _debug=False):
    """
    A simple disk-backed cache decorator, tailored for computing P_k.
    Computed func values are kept in memory, written back to disk on every cache miss.
    Data is compressed before writing to disk if compress_data.
    """
    _file_name = file_name
    desired_suffix = '.pkl.lzma' if compress_data else '.pkl'
    if not _file_name.endswith(desired_suffix):
        _file_name += desired_suffix

    def decorator(func):
        try:
            if compress_data:
                with lzma.LZMAFile(_file_name, 'rb') as infile:
                    cache = pickle.load(infile)
            else:
                with open(_file_name, 'rb') as infile:
                    cache = pickle.load(infile)
        except (IOError, ValueError, EOFError):
            cache = tuple()

        def save_func():
            if compress_data:
                with lzma.LZMAFile(_file_name, 'wb') as outfile:
                    pickle.dump(cache, outfile)
            else:
                with open(_file_name, 'wb') as outfile:
                    pickle.dump(cache, outfile)

        def new_func(ind):
            nonlocal cache, save_func
            try:
                ret = cache[ind]
                if _debug:
                    print("Cache hit:", ind)

                return ret

            except IndexError:
                if _debug:
                    print("Cache miss:", ind)

                # We need all the intermediate values till `ind`, since the P_k polynomials are
                # computed recursively, by [1, Eq. (2.24)].
                current_len = len(cache)
                for i in range(current_len, ind + 1):
                    if _debug:
                        print("Computing value", i)
                    cache = (*cache, func(i))

                save_func()

            return cache[ind]

        return new_func

    return decorator


# Compute each polynomial only once, caching the results.
# Use a disk-backed cache, since this is an *algebraic* definition.
@_cache_list_to_file(os.path.join(CACHE_BASEDIR, '.P_k_cache'))
def P_k(k: int):
    """
    The k-th polynomial P_k, [1, Eq. (2.23)-(2.24)].

    Returned as a SymPy expression. For calculating with numpy see P_k_for_numpy().
    Note that these are defined in purely algebraic terms, and so are *problem independent*.
    """
    assert k >= 0 and k == int(k), "Unexpected index value {}".format(k)
    if k == 0:
        # The zeroth polynomial, [1, Eq. (2.23)].
        return sympy.Integer(1)

    # k is at least 1.
    x0, x1 = sympy.symbols('x0 x1')
    P_k_minus_1 = P_k(k - 1)
    # The polynomials' recursive definition, [1, Eq. (2.24)].
    return sympy.expand((x1 - x0) * P_k_minus_1 + dbar_deriv(P_k_minus_1))


# A memory-only cache.
@functools.lru_cache(maxsize=None)
def P_k_for_numpy(k: int):
    """
    The k-th polynomial P_k, [1, Eq. (2.23)-(2.24)], in a form suited for numpy calculations.

    See P_k() for algebraic manipulations.
    """
    assert k >= 0 and k == int(k), "Unexpected index value {}".format(k)
    x = lambda n: sympy.symbols('x' + str(n))
    variables = tuple(x(n) for n in range(k + 1))
    return sympy.lambdify(variables, P_k(k), modules='numpy')


def _expected_distortion_power(D, direct_encoder, k: int) -> np.ndarray:
    """
    The expectation [1, Eq. (2.28)] with respect to X-hat of the k-th power of the distortion matrix D.

    The inputs D and direct_encoder should be indexed (t, x).
    Result is indexed (0, x). If k == 0 then D is returned.
    """
    assert k >= 0 and k == int(k), "Unexpected power " + str(k)
    if k == 0:
        return D

    # So k > 1. Matrices are indexed (t, x).
    return np.sum(direct_encoder * D**k, axis=0, keepdims=True)


def _padded_partition_vectors(max_n: int) -> np.ndarray:
    """
    Integer-partition vectors of 0...max_n (including), right-padded with zeros till length max_n.
    """
    assert max_n == int(max_n) >= 0
    yield np.zeros(max_n, dtype=np.uintc)
    for k in range(1, 1 + max_n):
        for p in root_tracking._partition_vectors(k):
            yield np.pad(p, (0, max_n - k))


def _integral_vectors(norm: int, coords: int) -> [int]:
    """
    Yield distinct integral vectors on coords coordinates with a specified norm (=sum of coordinates).
    """
    if coords == 1:
        yield (norm, )
        return

    for i in range(norm + 1):
        for v in _integral_vectors(norm - i, coords - 1):
            yield (i, ) + v


def _unique_permutations(sequence):
    """
    Yield unique permutations of input sequence.

        ('a', 'b', 'b')     -->     ('a', 'b', 'b'), ('b', 'a', 'b'), ('b', 'b', 'a')
    """
    # TODO: Replace by something more efficient?
    for seq in tuple(set(permutations(sequence))):
        yield seq


class IdMinusBADerivativeTensorsInMarginalCoordinates(root_tracking.AbstractCalcDerivativeTensors):
    """
    Implements the formulas of [1, Theorem 4] for the higher derivative tensors of the RD operator
    $Id - BA_beta$, [1, Equation (1.2)], in cluster-marginal coordinates.
    This is `Calc Deriv Tensor` when computing implicit derivatives with [1, Algorithm 1] for RD.

    Computations are done in three steps --- see also [1, Section II.5.5]:
        (1) problem-independent algebraic pre-computations, disk-cached in advance --- at P_k();
        (2) pre-computations per RD solution: the matrices G(k, a) of [1, Eq. (2.32)]; and
        (3) the RD derivative tensors of a particular order.

    Parameters:
        `solution_point`        The RD solution (distribution and beta) at which the tensors are to be evaluated.
        `problem_def`           An RD problem definition.
        `max_deriv_order`       The maximal order of implicit derivatives that can be computed using instance.
                                Higher values require more memory and pre-computations, so increase with care.
        `precompute_on_init`    Do the pre-computations (2) at constructor if True, lazy if False.
        `eps_for_cluster_marginal`
                                A lower threshold on cluster marginals, to ensure that not computing too
                                close to a cluster-vanishing bifurcation.
        `_debug`                Emit debug information if True.

    NOTE    Several caches are used to speed-up computations. Should memory be a concern,
            these can be cleared using the clear_caches() method.
    """
    def __init__(self, solution_point: RD_defs.RD_sol_point, problem_def: RD_defs.RD_problem,
                 max_deriv_order: int, precompute_on_init: bool = False,
                 eps_for_cluster_marginal: float = 0.0001, _debug: bool = False):
        """
        Set-up the machinery for computing higher RD derivative tensors at a point.

        See class documentation for details.
        """
        # Sanities
        assert isinstance(problem_def, RD_defs.RD_problem), \
            "Expecting an RD_problem instance, defining an RD problem. Got: {}".format(problem_def)
        assert isinstance(solution_point, RD_defs.RD_sol_point), \
            "Expecting an RD_sol_point, defining an RD solution at a point. Got: {}".format(solution_point)
        assert max_deriv_order == int(max_deriv_order) > 0, \
            "Please specify the maximal derivative order to be supported. Got: {}".format(max_deriv_order)
        self._debug, self._max_deriv_order = _debug, max_deriv_order

        # Since we're using cluster marginal coordinates, the problem has only T_dim
        # coordinates rather than the entire X_dim * T_dim coordinates of an RD problem.
        problem_dim = problem_def.T_dim

        # Delegate general-purpose setups.
        super().__init__(solution_point=solution_point, solution_beta=solution_point.beta,
                         problem_dim=problem_dim, problem_def=problem_def)

        assert 0 <= eps_for_cluster_marginal <= 1, \
            "Unexpected value {} for eps_for_cluster_marginal".format(eps_for_cluster_marginal)
        assert not (self.cluster_marginal < eps_for_cluster_marginal).any(), \
            "Cannot compute derivatives at near-zero cluster marginals {}".format(self.cluster_marginal)

        self._done_precomputations = False
        if precompute_on_init:
            self._perform_precomputations()

        # Caching should be per-instance, rather than a property of the class,
        # to avoid issues with garbage collection (memory issues).
        self.P_k_matrix = functools.lru_cache(maxsize=None)(self._P_k_matrix)
        self.calc_derivative_tensor = functools.lru_cache(maxsize=None)(self._calc_derivative_tensor)
        self._compute_direct_encoder = functools.lru_cache(maxsize=1)(self._compute_direct_encoder)

        if _debug:
            print("Done constructing an IdMinusBADerivativeTensorsInMarginalCoordinates "
                  "object {}, at beta={}".format(self, solution_point.beta))

    def precomputations_already_done(self):
        return self._done_precomputations

    def _perform_precomputations(self):
        """
        Pre-compute the G matrices [1, Eq. (2.32)], supporting derivative calculation to order max_deriv_order.
        """
        if self.precomputations_already_done():
            if self._debug:
                print("Already done pre-computations.")
            return

        if self._debug:
            print("Performing pre-computations to order {}.".format(self._max_deriv_order))

        # G [1, Eq. (2.32)] is indexed (k, a, t, x), with 0 <= k <= max_order, 0 <= a <= 1 + max_order.
        # See [1, Section II.5.5] for details.
        max_order, T_dim, X_dim = self._max_deriv_order, self.problem_def.T_dim, self.problem_def.X_dim
        G = np.zeros(shape=(1 + max_order, 2 + max_order, T_dim, X_dim))
        dtype = getattr(np, str(G.dtype))                   # An ugly hack to resolve numerical issues.
        factorial, vect_factorial = root_tracking.factorial, root_tracking._vector_factorial

        # Sum over (padded) partitions, till k_max = max_order.
        for t in _padded_partition_vectors(max_order):
            # The partitioned integer,
            k = np.sum(t * np.arange(1, 1 + max_order))
            assert k <= max_order, "Was expecting {} <= {}".format(k, max_order)

            # Compute the inner product to the right of [1, Eq. (2.32)].
            # The value of `a` can be determined later, when dividing by (a - |t|)! as `a` varies.
            P_k_prod = np.prod(np.array([(self.P_k_matrix(j) / factorial(j))**t[j-1]
                                         for j in range(1, 1 + k)]), axis=0) / vect_factorial(t)

            t_norm = np.sum(t)
            # The inner product `P_k_prod` contributes to G(k, a) at multiple `a` values.
            for a in range(t_norm, 2 + max_order):
                # 0 <= k <= max_order, 0 <= a <= 1 + max_order
                G[k, a, :, :] += dtype(P_k_prod / factorial(a - t_norm))

        # Set G(k, a) = 0 if a = 0 < k
        for k in range(1, 1 + max_order):
            G[k, 0, ...] = 0

        self._G = G
        self._done_precomputations = True
        if self._debug:
            print("Done precomputations.")

    def calc_derivative_tensor(self):
        """ A dummy method, replaced at __init__() with the cached version of _calc_derivative_tensor() below. """
        assert False, "This code should never be invoked."

    def _calc_derivative_tensor(self, beta_deriv_order: int, x_deriv_order: int) -> np.ndarray:
        """
        The higher derivative tensors of the RD operator in cluster-marginal coordinates, [1, Theorem 4].

        Compute the `x_deriv_order` derivative with respect to cluster-marginal
        coordinates, and `beta_deriv_order` derivative with respect to beta.
        A multi-dimensional tensor is returned; see documentation of base class AbstractCalcDerivativeTensors.

        This method is cached (see constructor).
        """
        # Delegate sanity checks.
        super().calc_derivative_tensor(beta_deriv_order=beta_deriv_order, x_deriv_order=x_deriv_order)
        assert x_deriv_order > 0 or beta_deriv_order > 0, "What is there to calculate?"

        if self._debug:
            print("Computing derivatives tensor, of orders ({}, {}) with "
                  "respect to (beta, marginals).".format(beta_deriv_order, x_deriv_order))

        # The real work starts here.
        if x_deriv_order == 0:
            # A derivative with respect to beta only --- this is [1, Eq. (2.30)]. Coordinates are indexed (t, x).
            p_x_as_row = RD_defs._make_column_vec(self.problem_def.p_x).T
            return -np.sum(p_x_as_row * self.direct_encoder * self.P_k_matrix(beta_deriv_order),
                           axis=1, keepdims=True)

        # Derive (also) with respect to the coordinates, [1, Eq. (2.31)]. Requires G to be pre-computed.
        if not self.precomputations_already_done():
            self._perform_precomputations()

        # Derivatives tensor has (1 + x_deriv_order) axes, each with problem_dim coordinates:
        #   one axis per derivative with respect to the coordinates,
        #   and one for the coordinates of the operator itself (zeroth axis).
        coords_in_axis, num_of_axes, X_dim = self.dim, 1 + x_deriv_order, self.problem_def.X_dim
        assert coords_in_axis == self.problem_def.T_dim == self.problem_dim, "This was not supposed to happen!"
        assert x_deriv_order <= self._max_deriv_order and beta_deriv_order <= self._max_deriv_order, \
            "Precomputations were done only to order {}, but a derivative of order ({}, {}) was " \
            "requested.".format(self._max_deriv_order, beta_deriv_order, x_deriv_order)

        deriv_tensor = np.zeros(shape=(coords_in_axis, ) * num_of_axes)
        factorial, vector_factorial = root_tracking.factorial, root_tracking._vector_factorial
        # Matrices are indexed (t, x).
        p_x_as_row = RD_defs._make_column_vec(self.problem_def.p_x).T
        encoder_over_marginal = self.direct_encoder / RD_defs._make_column_vec(self.cluster_marginal)

        # A multi-index alpha_plus=(alpha_1, alpha_2, ..., alpha_M) at [1, Eq. (2.31)] represents the
        # number of differentiations with respect to each of the coordinates. e.g., alpha_1 differentiations
        # with respect to x_1, till alpha_M differentiations with respect to x_M.
        # To populate the coordinates of the derivatives tensor at [1, Eq. (2.31)], we need to iterate over
        # all the multi-indices alpha_plus. To each multi-index there may correspond multiple tensor
        # entries --- see [1, Eq. (2.6)], and especially the comments following it on indexation.

        # Iterate over multi-indices, whose norm is the number of times we derive with respect to the coordinates.
        for alpha_plus in _integral_vectors(norm=x_deriv_order, coords=coords_in_axis):
            tensor_coords_ordered = [ind for ind, val in enumerate(alpha_plus) for _ in range(val)]
            # Following the comments above, tensor_coords_ordered is a particular nicely-ordered enumeration
            # of the derivatives' tensor coordinates, corresponding to alpha_plus:
            # tensor_coords[i] enumerates the (i + 1)-th coordinate of the derivatives tensor,
            # while alpha_plus[i] counts how many times do we derive with respect to the i-th
            # coordinate. e.g., for 2 axes (2+1 counting the first) with 4 coordinates in each,
            #       tensor_coords:          alpha_plus:
            #       (0, 0)                  (2, 0, 0, 0)          <--- alpha_plus sums up to the number of axes,
            #       (0, 1)                  (1, 1, 0, 0)               which is nothing but the number of
            #       (1, 3)                  (0, 1, 0, 1)               differentiations with respect to the coordinates.
            # To each alpha_plus there usually corresponds multiple tensor_coords vectors.
            # We compute the above derivative only once, but assigned it multiple times, below.
            summation_integrand = np.zeros(shape=(coords_in_axis, X_dim))
            # Compute the integrand of the inner summation at [1, Eq. (2.31)], per multi-index alpha_plus.
            for k in _integral_vectors(norm=beta_deriv_order, coords=coords_in_axis):
                # The inner product there is over cluster numbers `t`.
                for t in range(coords_in_axis):
                    # G is indexed (k, a, t, x), with 0 <= k <= max_order, 0 <= a <= 1 + max_order.
                    stacked = np.vstack([self._G[k[t], alpha_plus[t], t, :] for t in range(coords_in_axis)])

                    stacked[t, :] = alpha_plus[t] * self._G[k[t], alpha_plus[t], t, :] - \
                                    x_deriv_order * (1 + alpha_plus[t]) * self.direct_encoder[t, :] * \
                                    self._G[k[t], 1 + alpha_plus[t], t, :]

                    # Accumulate over inner products.
                    summation_integrand[t, :] += np.prod(stacked, axis=0, keepdims=False)

            # The leftmost term at [1, Eq. (2.31)].
            delta_term = (beta_deriv_order == 0) * (x_deriv_order == 1) * np.array(alpha_plus)

            # Coefficients of the second term at [1, Eq. (2.31)].
            alpha = (beta_deriv_order, ) + alpha_plus
            assert x_deriv_order == np.sum(alpha_plus)
            # Probability distributions are indexed (t, x).
            coef = (-1)**(x_deriv_order - 1) * factorial(x_deriv_order - 1) * vector_factorial(alpha)
            alpha_plus_as_col = RD_defs._make_column_vec(np.array(alpha_plus))
            encoder_over_marginal_to_alpha = np.prod(encoder_over_marginal ** alpha_plus_as_col, axis=0)
            # Indexed by t.
            deriv_vec_value = delta_term - \
                              coef * np.sum(p_x_as_row * encoder_over_marginal_to_alpha * summation_integrand,
                                            axis=1, keepdims=False)

            assert len(deriv_vec_value) == coords_in_axis
            # Distribute the computed values to the corresponding tensor coordinates ---
            # see indexation-related comments around the beginning of the main `for` loop.
            for tensor_coords in _unique_permutations(tensor_coords_ordered):
                deriv_tensor[(..., *tensor_coords)] = deriv_vec_value

        if self._debug:
            print("Done.")

        return deriv_tensor

    @property
    def direct_encoder(self):
        """ The direct-encoder at the point of evaluation. RD-specific, indexed (t, x). """
        return self._compute_direct_encoder()

    def _compute_direct_encoder(self) -> np.ndarray:
        """ Computes the direct encoder; method cached (see constructor). """
        return self.point.direct_encoder(self.problem_def)

    @property
    def cluster_marginal(self) -> np.ndarray:
        """ An alias to the cluster marginal at the point of evaluation. RD-specific. """
        return self.point.p_t

    @property
    def p_t(self) -> np.ndarray:
        """ An alias to the cluster marginal at the point of evaluation. RD-specific. """
        return self.point.p_t

    @property
    def solution_point_as_vector(self) -> np.ndarray:
        """ An alias to the cluster marginal at the point of evaluation. RD-specific. """
        return self.cluster_marginal

    def _P_k_matrix(self, k: int) -> np.ndarray:
        """ The P_k matrices, defined by [1, Eq. (2.29)]; method cached (see constructor)."""
        # TODO: Cache _expected_distortion_power()?
        return P_k_for_numpy(k)(*[_expected_distortion_power(self.problem_def.D, self.direct_encoder, n)
                                  for n in range(k + 1)])

    def __del__(self):
        self.clear_caches()

    def clear_caches(self):
        """
        Clear caches pertaining to *this* object.

        Does not clear the caches of P_k_for_numpy() and P_k(), as they do *not* pertain to a particular RD solution.
        """
        for attr in ['P_k_matrix', 'calc_derivative_tensor']:
            # Function might be called during destruction, so take care when accessing attributes.
            if hasattr(self, attr) and hasattr(getattr(self, attr), 'cache_clear'):
                getattr(self, attr).cache_clear()

        # TODO: Clear also cache of direct_encoder?
        if hasattr(self, '_G'):
            del self._G

        self._done_precomputations = False
        if hasattr(self, '_debug') and self._debug:
            print("Caches cleared for {}.".format(self))
