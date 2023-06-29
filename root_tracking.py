"""
Implements [1, Algorithm 1] for computing the implicit multivariate derivatives

    d^l x / d beta^l        [1, Eq. (1.4)]

at a root of an arbitrary operator F,

    F(x, beta) = 0 .        [1, Eq. (1.3)]

The root's trajectory can then be extrapolated, as in [1, Eq. (1.5)].

[1, Algorithm 1] for implicit derivatives is implemented by the calculate_implicit_derivative() method of
the RootTracker class. It requires an auxiliary `Calc Deriv Tensor` method, for the derivative tensors of F.
This auxiliary method is represented by the abstract base class AbstractCalcDerivativeTensors.
One can specialize this to a particular setting (e.g., RD) by deriving the abstract base class.
`Calc Deriv Tensor` is then to be implemented by overriding the calc_derivative_tensor() method of the derived class.

Example
    The specialization of Algorithm 1 to RD is provided at the
    IdMinusBADerivativeTensorsInMarginalCoordinates class, in RD_root_tracking.py.

Note
    This module does was intentionally written as a standalone, and does *not* depend on RD code.
    In principle, it can be used to compute higher multivariate implicit derivatives for
    an arbitrary operator F.
"""
from scipy.special import factorial
import numpy as np
from abc import ABC, abstractmethod
from collections import Counter


class RootTrackingError(Exception):
    """ A dummy class for nicer Exceptions. """
    pass


class AbstractCalcDerivativeTensors(ABC):
    """
    An abstract base class for the derivative tensors of an operator F(x, beta),
    at its approximate roots. See module's help for problem settings.

    Parameters:
        `problem_dim`       the problem's dimension;
            i.e., the dimension of x, with F(x, beta) an operator on R^`problem_dim`.
        `solution_point`    the point x at which the derivative tensors are to be evaluated.
        `solution_beta`     the beta value at which the derivative tensors are to be evaluated.
        `problem_def`       additional domain-specific parameters, such as an RD problem definition.

    The derivative tensors of F with the above parameters are to be implemented by the
    calc_derivative_tensor() method at a derived class; this is `Calc Deriv Tensor` of [1, Algorithm 1].

    Since F has `problem_dim` coordinates, its k-fold derivative with respect to the coordinates x is a
    (1 + k)-tensor (a "matrix" with k+1 axes), each axis with `problem_dim` coordinates. In contrast, the
    dimensions of the derivative tensor do *not* depend on the number of differentiations with respect to beta.
    See more details around [1, Eq. (2.6) in Section 2.1].

    Usage:
        See class IdMinusBADerivativeTensorsInMarginalCoordinates in RD_root_tracking.py.
    """
    def __init__(self, solution_point, solution_beta: float, problem_dim: int, problem_def=None):
        """ Set-up the machinery for calculating an operator's derivative tensors. See class docstring for details. """
        assert problem_dim == int(problem_dim) > 0, "Unexpected problem dimension {}".format(problem_dim)
        self.dim = problem_dim
        assert solution_point, "Must have *some* point at which the operator is " \
                               "to be evaluated: {}".format(solution_point)
        self.point = solution_point
        assert solution_beta is not None
        self.beta = solution_beta
        # Any further problem definition details go here.
        self.problem_def = problem_def

    @abstractmethod
    def calc_derivative_tensor(self, beta_deriv_order: int, x_deriv_order: int):
        """
        An abstract method for evaluating an operator's derivative tensors, `Calc Deriv Tensor` of [1, Algorithm 1].

        To be implemented for a particular operator F by overriding method in a derived class.

        Parameters:
            `x_deriv_order`     the derivative's order with respect to the coordinates x.
            `beta_deriv_order`  the derivative's order with respect to the time-like parameter beta.

        Returned tensor has (1 + x_deriv_order) axes, each with problem_dim coordinates.
        The first axis corresponds to the coordinates of F.
        """
        assert x_deriv_order == int(x_deriv_order) >= 0, \
            "Unexpected derivation order {} with respect to x".format(x_deriv_order)
        assert beta_deriv_order == int(beta_deriv_order) >= 0, \
            "Unexpected derivation order {} with respect to beta".format(beta_deriv_order)

    @property
    def problem_dim(self) -> int:
        """ The problem's dimension. """
        return self.dim

    @property
    def solution_point(self):
        """ The point of evaluation. """
        return self.point

    @property
    def solution_beta(self) -> float:
        """ The beta value of evaluation. """
        return self.beta

    @property
    @abstractmethod
    def solution_point_as_vector(self):
        """ A vectorized form of the solution_point property, to be overridden by a derived class. """
        pass


class RootTracker:
    """
    Methods for tracking roots of an operator equation F(x, beta) = 0, as in [1, Eq. (1.3)].

    Main methods:
        calculate_implicit_derivative()
            Implements [1, Algorithm 1], for the implicit multivariate derivatives at an operator's root.

        taylor_approx()
            The taylor polynomial (in beta), computed at the current root.

    Parameters:
        `deriv_tensor_calc`     An auxiliary object calculating the derivative tensors of F.
                                Must be an instance of a class derived from AbstractCalcDerivativeTensors.
    """
    def __init__(self, deriv_tensor_calc: AbstractCalcDerivativeTensors, _debug: bool = False):
        assert isinstance(deriv_tensor_calc, AbstractCalcDerivativeTensors), \
            "Need an object capable of computing high-order derivative tensors at a point."
        self._deriv_tensor_calc = deriv_tensor_calc

        assert isinstance(_debug, bool)
        self._debug = _debug
        if _debug:
            print("Initializing a RootTracker instance at {}.".format(str(deriv_tensor_calc)))

        # Cache implicit derivatives. Derivative of order (d + 1) is to be stored at _deriv_cache[d].
        self._deriv_cache = list()

        # Inverse-Jacobian is used once per derivative order, so cache it.
        J = deriv_tensor_calc.calc_derivative_tensor(beta_deriv_order=0, x_deriv_order=1)
        try:
            self._J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError as e:
            raise RootTrackingError("A singular Jacobian is not yet handled by this code. "
                                    "Can either extend code if you know what you're doing, or "
                                    "(better) double-check why your Jacobian was singular in "
                                    "the first place (for RD it shouldn't be). Got:\n{}".format(J)) from e

    def calculate_implicit_derivative(self, deriv_order: int) -> np.ndarray:
        """
        The implicit derivatives of order `deriv_order` at the given root; this is [1, Algorithm 1].

        Return a derivatives vector.
        """
        assert deriv_order == int(deriv_order) >= 1, "Unexpected derivative order {}.".format(deriv_order)

        # Use cached results.
        if len(self._deriv_cache) >= deriv_order:
            if self._debug:
                print("Implicit derivative of order {} already exists in cache, returning.".format(deriv_order))

            # Note zero-based indexation of Python lists!
            return self._deriv_cache[deriv_order - 1]

        # All the lower-order derivatives are needed to compute this one.
        if len(self._deriv_cache) < deriv_order - 1:
            if self._debug:
                print("Need lower-order implicit derivatives to calculate that of "
                      "order {}; computing recursively.".format(deriv_order))

            # Get lower order implicit derivatives: lines 3-4 in [1, Algorithm 1].
            self.calculate_implicit_derivative(deriv_order=deriv_order - 1)

        if self._debug:
            print("Computing implicit derivative of order {}.".format(deriv_order))

        tensor_calc, problem_dim = self._deriv_tensor_calc, self.problem_dim
        new_deriv = np.zeros(problem_dim)
        dtype = getattr(np, str(new_deriv.dtype))           # An ugly hack to resolve numerical issues.

        # Iterate all partition vectors, written by multiplicity: line 6 in [1, Algorithm 1].
        for m_vect in _partition_vectors(deriv_order):
            # The partition has parts p_vect, of multiplicities m_vect.
            p_vect = np.arange(1, 1 + deriv_order)      # Part sizes
            M = np.sum(m_vect)                          # Total multiplicity: line 9 in [1, Algorithm 1].
            p_vect_factorial = factorial(p_vect)

            # Multiplicity of the part of size 1: line 7 in [1, Algorithm 1].
            b_max = m_vect[0]
            # Inner summation, on the number of differentiations with respect to beta: line 8 in [1, Algorithm 1].
            for b in range(0, 1 + b_max):
                # Ignore the trivial partition, which corresponds to the Jacobian matrix: line 10 in [1, Algorithm 1].
                if M == 1 and b == 0:
                    # This is the trivial partition. See [1, Corollary 2] and the comments preceding it.
                    assert _is_trivial_partition_vector(m_vect), \
                        "Weren't we expecting a trivial partition? Multiplicities vector = {}".format(m_vect)
                    continue

                # Compute summand's coefficient: line 13 in [1, Algorithm 1].
                padded_m_vect = np.pad(m_vect, (1, 0))
                padded_m_vect[:2] = [b, m_vect[0] - b]
                # TODO: Improve handling of large numerators & denominators.
                coef = factorial(deriv_order) / (_vector_factorial(padded_m_vect) * np.prod(p_vect_factorial ** m_vect))
                assert coef == int(coef) > 0

                # Get the corresponding derivatives tensor: line 14 in [1, Algorithm 1].
                # Must *not* use the cached derivative tensors directly, without the copy() instruction,
                # as this modifies cached values and leads to bugs.
                deriv_tensor = tensor_calc.calc_derivative_tensor(beta_deriv_order=b, x_deriv_order=M-b).copy()

                # Multi-linear products, at the rightmost term of line 15 in [1, Algorithm 1].
                _m_vect = m_vect.copy()
                # Account for differentiations with respect to beta,
                _m_vect[0] -= b
                # Index axes number:
                axis_ind = 1
                # Iterate over part sizes and their multiplicities to form the
                # multi-linear product, per RHS of [1, Eq. (2.17)].
                for part_size_, part_multiplicity in enumerate(_m_vect):
                    # Note Python's zero-based indexing.
                    part_size = 1 + part_size_
                    # The multiplicity determines along how many axes the
                    # implicit derivative of order `part_size` should be placed.
                    for t in range(part_multiplicity):
                        selected_deriv_vect = self._deriv_cache[part_size - 1].flatten()
                        # Derivatives vector of appropriate order is multiplied along `axis_ind` axis.
                        deriv_vec_expanded = np.expand_dims(selected_deriv_vect,
                                                            tuple(range(axis_ind)) + tuple(range(axis_ind + 1, M-b+1)))
                        assert deriv_vec_expanded.shape[axis_ind] == problem_dim
                        assert deriv_vec_expanded.ndim == deriv_tensor.ndim
                        deriv_tensor *= deriv_vec_expanded
                        axis_ind += 1

                # Should have iterated along the `M - b` axes of the derivatives tensor (+1 for the coordinates of F).
                assert axis_ind == 1 + M - b

                # Reduce the multi-linear product, at the rightmost term of line 15 in [1, Algorithm 1].
                # Zeroth axis are the coordinates of F --- _not_ to be summed over.
                reduced_tensor = np.sum(deriv_tensor, axis=tuple(range(1, 1 + M - b)))

                # Add reduced tensor to running sum: subtraction at line 15 in [1, Algorithm 1].
                new_deriv -= dtype(coef * reduced_tensor.flatten())

        # Linear pre-image under Jacobian: line 20 in [1, Algorithm 1].
        new_deriv = np.matmul(self._J_inv, new_deriv)
        self._deriv_cache.append(new_deriv)
        if self._debug:
            print("Done calculating derivative of order {}.".format(deriv_order))

        return new_deriv

    @property
    def problem_def(self):
        """ The problem's definition or paramters. """
        return self._deriv_tensor_calc.problem_def

    @property
    def problem_dim(self) -> int:
        """ The problem's dimension. """
        return self._deriv_tensor_calc.dim

    @property
    def solution_point(self):
        """ The point of evaluation. """
        return self._deriv_tensor_calc.point

    @property
    def solution_beta(self) -> float:
        """ The beta value of evaluation. """
        return self._deriv_tensor_calc.beta

    def taylor_coefficients(self, degree: int) -> np.ndarray:
        """
        Taylor coefficients (matrix) of the root's beta-expansion by problem's coordinate.

        Returned matrix has the d-th Taylor coefficient of the problem's i-th coordinate at its [d, i] entry.

        See also taylor_approx().
        """
        # The k-th order monomial in the Taylor expansion is comprised of,
        #   - The k-th derivative,
        #   - The k-th power of the deviation $\Delta \beta$ from $\beta_0$,
        #   - And the combinatorial coefficient at the denominator.
        assert degree >= 0, "Unexpected degree {} for a Taylor polynomial.".format(degree)
        return np.vstack([self._deriv_tensor_calc.solution_point_as_vector.flatten(), ] +
                         [self.calculate_implicit_derivative(i) / factorial(i) for i in range(1, 1 + degree)])

    def taylor_approx(self, order: int) -> [np.poly1d]:
        """
        A list of np.poly1d's, one per problem coordinate, for the Taylor expansion [1, Eq. (1.5)] around the given root.
        """
        taylor_coeffs = self.taylor_coefficients(order)
        polys = [np.poly1d(taylor_coeffs.T[i, ::-1]) for i in range(self._deriv_tensor_calc.problem_dim)]

        # Sanity: expansion at beta_0 should be given base-point.
        p_0 = self._deriv_tensor_calc.solution_point_as_vector.flatten()
        assert (np.array([p(0) for p in polys]) == p_0).all()
        assert len(polys) == self.problem_dim
        return polys
        # TODO: Replace np.poly1d with np.polynomial.polynomial.Polynomial ?


def _partitions(n: int) -> [int]:
    """
    Yield partitions of the integer n.

        3       -->     (1, 1, 1), (1, 2), (3,)
    """
    # 0 has only the empty one.
    if n == 0:
        yield tuple()
        return

    # Construct partitions of n using those of n-1.
    for p in _partitions(n - 1):
        # Can either prepend a `1` to a partition of (n-1),
        yield (1, ) + p
        # or increment its first coordinate (if possible).
        if p and (len(p) < 2 or         # p has just a single coordinate (a trivial partition).
                  p[0] < p[1]):         # There are >= 2 coordinates, and can increment first.
            yield (p[0] + 1, ) + p[1:]


def _partition_vectors(n: int) -> np.ndarray:
    """
    Yield partition vectors of n, representing partition's multiplicity.

    Represented by multiplicity, a partition of n is a vector of length n,
    with coordinate k representing the multiplicity of the
    integer (k + 1) in the partition. e.g., for n=3,
        Partitions:                 Partition vectors:
        3               <~~~>       (0, 0, 1)
        2 + 1           <~~~>       (1, 1, 0)
        1 + 1 + 1       <~~~>       (3, 0, 0)
    """
    # TODO: Replace with a faster implementation.
    for p in _partitions(n):
        vec = np.zeros(shape=n, dtype=np.uintc)
        for k, v in Counter(p).items():
            vec[k - 1] = v

        yield vec


def _is_trivial_partition_vector(vec: [int]) -> bool:
    """ Return True iff the vector vec represents the multiplicities of a trivial partition. """
    return vec[-1] == 1 and (vec[:-1] == 0).all()


def _vector_factorial(vec: np.ndarray) -> float:
    """ The factorial of a vector, defined as the product of entry-wise factorials; e.g., [1, Eq. (4.3)]. """
    # TODO: warn on inexact result?
    with np.errstate(all='raise'):
        return np.prod(factorial(vec))

