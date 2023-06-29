"""
Methods for solving RD problems numerically. In particular, implements BA's algorithm [2] and RTRD [1].

For starter, use solve_RD_probs() to dispatch the computation of an RD problem with a supported algorithm.

Noteworthy functions:
    solve_RD_probs()            A convenience dispatcher, invoking the algorithm defined by the computation parameters.
    BA_until_convergence()      Blahut's algorithm [2] at a given beta value.
    BA_solver()                 Solve along a grid using BA, either independently or with reverse annealing.
    RTRD_fixed_order_and_step_size()
                                Root-tracking for RD, [1, Algorithm 3], with fixed order and step sizes.
    extrapolate_intermediate_RTRD_solutions()
                                Extrapolate off-grid solutions from those obtained by RTRD.
    exact_solutions_for_binary_source_with_Hamming_distortion()
                                Exact solutions and derivatives for a binary source with a Hamming distortion measure.

Usage
    See README.md
"""
import traceback
import warnings
from collections import deque
from copy import copy
from datetime import datetime
from collections.abc import Iterable
import numpy as np
from numpy.random import default_rng
import sympy
import RD_defs
import RD_root_tracking
import root_tracking
from RD_defs import debug_print, log2_xrange_domain

rng = default_rng()


def solve_RD_probs(probs, compute_params):
    """ Solve an RD problem with the algorithm defined by its `compute_params` (`computation_method`). """
    # For 'reverse annealing', 'independent' and RTRD, respectively.
    solver_funcs = (BA_solver, BA_solver, RTRD_fixed_order_and_step_size)

    solver_func = lambda compute_params: \
        solver_funcs[RD_defs.ADMISSIBLE_COMPUTATION_METHODS.index(compute_params.computation_method)]

    start = datetime.now()
    multiple_problems = isinstance(probs, Iterable) and isinstance(compute_params, Iterable)
    if not multiple_problems:
        sols = solver_func(compute_params)(probs, compute_params)
        debug_print("Computation's duration: {}".format(str((datetime.now() - start))))
        return sols

    sols_list = [solver_func(compute_params)(prob, params) for prob, params in zip(probs, compute_params)]
    debug_print("Computation's duration: {} (for all problems)".format(str((datetime.now() - start))))
    return sols_list


def _check_basic_sanities(prob, params, check_log2_beta_range=True):
    assert isinstance(prob, RD_defs.RD_problem), "Expecting an `RD_problem` instance, got: {}".format(prob)
    assert isinstance(params, RD_defs.RD_compute_params), \
        "Expecting an `RD_compute_params` instance, got: {}".format(params)
    if check_log2_beta_range:
        assert params.log2_beta_range is not None, \
            "Expecting a range of beta values, got: {}".format(str(params.log2_beta_range))


def _single_BA_iteration(current_cluster_marginal: np.ndarray, p_x: np.ndarray,
                         exp_minus_beta_d: np.ndarray) -> np.ndarray:
    """ A single BA iteration in marginal coordinates, as in [1, Eq. (2.20)]. """
    direct_enc = \
        current_cluster_marginal * exp_minus_beta_d / np.sum(current_cluster_marginal * exp_minus_beta_d,
                                                             axis=0, keepdims=True)
    return np.matmul(direct_enc, p_x)


def BA_until_convergence(prob: RD_defs.RD_problem, beta: float,
                         compute_params: RD_defs.RD_compute_params,
                         initial_cluster_marginal: np.ndarray = None) -> RD_defs.RD_sol_point:
    """
    Blahut's algorithm [2].

    Return an RD_sol_point instance, representing the obtained solution.

    Parameters:
        `prob`              RD problem definition.
        `beta`              Lagrange multiplier value beta.
        `compute_params`    Parameters used for computation: `uniform_initial_conditions`,
                            `max_BA_iters_allowed` and `BA_stopping_condition`.
        `initial_cluster_marginal`
                            BA's initialization. If None, then use either a uniform initial
                            condition, or a randomly-generated one (depending on compute_params).
    """
    _check_basic_sanities(prob, compute_params, check_log2_beta_range=False)
    assert beta > 0, "Expecting beta > 0, got: {}".format(beta)

    if initial_cluster_marginal is not None:
        prev_cluster_marginal = RD_defs._make_column_vec(initial_cluster_marginal)
    else:
        # Generate an initial condition.
        if compute_params.uniform_initial_conditions:
            prev_cluster_marginal = np.ones(shape=(prob.T_dim, 1)) / prob.T_dim
        else:
            prev_cluster_marginal = rng.dirichlet([1]*prob.T_dim, size=1).T

    assert RD_defs._is_probability_distribution(prev_cluster_marginal), \
        "Initial condition should be a normalized distribution: {}".format(prev_cluster_marginal)

    exp_minus_beta_d = np.exp(-beta * prob.D)
    iter_count, solution_converged = 0, False

    while iter_count < compute_params.max_BA_iters_allowed and not solution_converged:
        cluster_marginal = _single_BA_iteration(prev_cluster_marginal, prob.p_x, exp_minus_beta_d)

        # L_infty stopping condition.
        solution_converged = \
            RD_defs._is_equal(prev_cluster_marginal, cluster_marginal, eps=compute_params.BA_stopping_condition)

        iter_count += 1
        prev_cluster_marginal = cluster_marginal

    return RD_defs.RD_sol_point(p_t=cluster_marginal, beta=beta, iter_count=iter_count)


def BA_solver(prob: RD_defs.RD_problem, compute_params: RD_defs.RD_compute_params) -> [RD_defs.RD_sol_point]:
    """
    Solve using Blahut's algorithm [2] along a grid, with either independent or
    reverse annealing initial conditions.
    """
    _check_basic_sanities(prob, compute_params, check_log2_beta_range=True)

    assert compute_params.sample_num is not None, "Please specify number of grid-points in `sample_num`."
    assert 'reverse annealing' in RD_defs.ADMISSIBLE_COMPUTATION_METHODS \
           and 'independent' in RD_defs.ADMISSIBLE_COMPUTATION_METHODS
    assert compute_params.computation_method in ('reverse annealing', 'independent'), \
        "This function only implements a reverse annealing and an independent sampling solver."
    reverse_annealing = (compute_params.computation_method == 'reverse annealing')

    debug_print("Starting BA computation. RD problem: {}\nComputation parameters: {}".format(prob, compute_params))

    beta_iterator = log2_xrange_domain(log_start=compute_params.log2_beta_range[0],
                                       log_stop=compute_params.log2_beta_range[1],
                                       num=compute_params.sample_num,
                                       inverse_order=True)

    solutions, last_cluster_marginal = deque(), None
    for beta in beta_iterator:
        new_sol = BA_until_convergence(prob=prob, beta=beta, compute_params=compute_params,
                                       initial_cluster_marginal=last_cluster_marginal)
        last_cluster_marginal = new_sol.p_t if reverse_annealing else None
        solutions.append(new_sol)

    debug_print("Done computing.")
    return sorted(solutions, key=lambda s: s.beta)


def _track_RD_root_to_bifurcation(prob: RD_defs.RD_problem,
                                  initial_sol: RD_defs.RD_sol_point,
                                  order: int, step_size: float,
                                  cluster_mass_threshold: float, _debug: bool = False) -> [RD_defs.RD_sol_point]:
    """
    Implements [1, Algorithm 2]: a modified Taylor method for RD with fixed orders and step-size.
    Tracks an RD root till the next cluster-vanishing bifurcation.

    Return a list of computed solutions in decreasing beta order, *including* initial_sol.
    The approximating polynomials computed at each solution are saved at its 'extrapolation_funcs' property.

    Except for the last solution, every marginal coordinate in the returned solutions is >= cluster_mass_threshold.
    """
    assert order == int(order) > 0, "Unexpected order {} for Taylor method.".format(order)
    assert step_size > 0, "Unexpected step size {} for Taylor method.".format(step_size)
    assert 0. < cluster_mass_threshold < .1, \
        "Unexpected cluster mass threshold {}, for bifurcation detection.".format(cluster_mass_threshold)
    assert initial_sol.beta > 0 and (initial_sol.p_t > cluster_mass_threshold).all(), \
        "Unexpected initial solution: {}".format(initial_sol)

    solutions, last_sol = deque(), initial_sol
    # Line 3 at [1, Algorithm 2]: Stop if too close to a bifurcation.
    while (last_sol.p_t > cluster_mass_threshold).all() and last_sol.beta > 0:
        # Lines 4-6 at [1, Algorithm 2]: compute implicit derivatives at `last_sol`, using [1, Algorithm 1] for RD.
        try:
            # Set-up the machinery for implicit derivatives in RD:
            #   RD derivative tensors, and a RootTracker instance using it to compute implicit derivatives.
            deriv_tensor_calc = \
                RD_root_tracking.IdMinusBADerivativeTensorsInMarginalCoordinates(solution_point=last_sol,
                                                                                 problem_def=prob,
                                                                                 max_deriv_order=order,
                                                                                 _debug=_debug)
            tracker = root_tracking.RootTracker(deriv_tensor_calc=deriv_tensor_calc, _debug=_debug)

            # Line 7 at [1, Algorithm 2]: compute Taylor expansions.
            taylor_polys = tracker.taylor_approx(order=order)
            deriv_tensor_calc.clear_caches()

            # Remember computed polynomials, so that we could later extrapolate off-grid points.
            last_sol.extrapolation_funcs = taylor_polys
            solutions.append(last_sol)

            # Line 8 at [1, Algorithm 2]: extrapolate the next grid-point.
            last_sol = RD_defs.RD_sol_point(p_t=np.array([p(-step_size) for p in taylor_polys]),
                                            beta=last_sol.beta - step_size)

        except AssertionError as e:
            debug_print("Computation failed at beta={:0.3g}; halting branch-tracking. "
                        "Exception: {}".format(last_sol.beta, traceback.format_exception_only(type(e), e)))

            if RD_defs._DEBUG:
                import pdb; pdb.pm()

            break

    # Put also last solution in returned list:
    #   either the one last extrapolated, or `initial_sol` if started too close to a bifurcation.
    solutions.append(last_sol)

    return list(solutions)


def RTRD_fixed_order_and_step_size(prob: RD_defs.RD_problem, compute_params: RD_defs.RD_compute_params,
                                   _debug: bool = False) -> [RD_defs.RD_sol_point]:
    """
    Implements root-tracking for RD, [1, Algorithm 3]:

    A modified Taylor method with fixed order and step-sizes between bifurcations,
    with a heuristic to detect and handle cluster-vanishing bifurcations.
    """
    _check_basic_sanities(prob, compute_params, check_log2_beta_range=True)
    if compute_params.sample_num is not None:
        debug_print("WARNING: sample_num set to {}, but is *ignored* by this method.".format(compute_params.sample_num))
    assert compute_params.computation_method == 'diff-eq fixed order and step size', \
        "This function only implements a vanilla Taylor method ODE-based solver."

    order, step_size, cluster_mass_threshold = \
        compute_params.order, compute_params.step_size, compute_params.cluster_mass_threshold
    assert order == int(order) > 0, "Unexpected derivatives' order: {}".format(order)
    assert 0 < step_size, "Unexpected step size: {}.".format(step_size)
    assert 0. < cluster_mass_threshold < .1, "Unexpected cluster mass threshold: {}".format(cluster_mass_threshold)

    debug_print("Starting RTRD computation, using *fixed* order and step-size. Problem: {}".format(prob))
    debug_print("Computation parameters: {}".format(compute_params))

    # Obtain an initial condition using BA.
    last_sol = BA_until_convergence(prob=prob, beta=2**compute_params.log2_beta_range[1],
                                    compute_params=compute_params)
    debug_print("Done {} BA iterations to obtain an initial solution at "
                "beta={:0.4g}.".format(last_sol.iter_count, last_sol.beta))
    reduced_prob, solutions, beta = copy(prob), deque(), last_sol.beta

    def expand_back_sol_dims(reduced_sol, clusters_with_mass):
        """
        Zero-pad marginal at missing clusters, expanding solution vector to its original dimension.

        `clusters_with_mass` is a logical array: True at coordinates which are in use by `reduced_sol`, False
        at coordinates which should be zero-padded. Return a copy of `reduced_sol`, zero-padded as necessary.
        """
        upscaled_p_t = np.zeros(len(clusters_with_mass))
        upscaled_p_t[clusters_with_mass] = reduced_sol.p_t.flatten()
        upscaled_sol = copy(reduced_sol)
        # Ensure the shape of `reduced_sol.p_t` is preserved.
        upscaled_sol.p_t = np.expand_dims(upscaled_p_t, np.where(np.array(reduced_sol.p_t.shape) == 1))
        # Zero-pad functions used for extrapolation.
        if hasattr(reduced_sol, 'extrapolation_funcs'):
            # Put the original extrapolation-function at coordinates used by the reduction,
            # and the zero polynomial at coordinates which were not.
            upscaled_sol.extrapolation_funcs = \
                [reduced_sol.extrapolation_funcs[np.sum(clusters_with_mass[:1 + i]) - 1] if val  # val is True / False
                 else np.poly1d(0) for i, val in enumerate(clusters_with_mass)]
        return upscaled_sol

    support_size = lambda sol: np.sum(sol.p_t > cluster_mass_threshold)
    # Line 2 at [1, Algorithm 3]: Stop if solution has a trivial support.
    while support_size(last_sol) > 1 and beta > 0:
        debug_print("Tracking till bifurcation: starting at beta={:0.4g} on the problem reduced to {} "
                    "clusters.".format(beta, support_size(last_sol)))

        # Line 3 at [1, Algorithm 3]: reduce problem and root to its support.
        clusters_with_mass = (last_sol.p_t.flatten() > cluster_mass_threshold)
        reduced_prob.D = prob.D[clusters_with_mass, :]          # D is indexed (t, x).
        # The root should be normalized, unless the initial condition happens to be too close to a bifurcation.
        last_sol.p_t = last_sol.p_t[clusters_with_mass] / np.sum(last_sol.p_t[clusters_with_mass])

        # Line 4 at [1, Algorithm 3]: Track solutions between bifurcations.
        sols_between_bifs = \
            _track_RD_root_to_bifurcation(prob=reduced_prob, initial_sol=last_sol, order=order,
                                          step_size=step_size, cluster_mass_threshold=cluster_mass_threshold,
                                          _debug=_debug)
        # Returned solutions list `sols_between_bifs` includes `initial_sol`.

        # Returned list `sols_between_bifs` should at least contain initial_sol.
        assert len(sols_between_bifs) > 1, \
            "Was expecting *some* solutions to have been computed, but got none."

        # Line 5 at [1, Algorithm 3]: Append items in `sols_between_bifs` to results, except for the last.
        for sol in sols_between_bifs[:-1]:
            # The solutions are in reduced coordinates; but computers don't like to work with vectors of
            # variable sizes, so expand back to the problem's original dimension, zero-padding as necessary.
            solutions.append(expand_back_sol_dims(sol, clusters_with_mass))

        # Line 6 at [1, Algorithm 3]: proceed with the last item in `sols_between_bifs`.
        last_sol = sols_between_bifs[-1]

        if last_sol.beta <= 0:
            # Cannot use BA (below) with negative beta values.
            debug_print("Target beta={:0.4g} is negative, stopping calculation early.".format(last_sol.beta))
            break

        # Lines 7-8 at [1, Algorithm 3]: erase nearly-vanished coordinates and normalize.
        new_marginal = last_sol.p_t.flatten()
        vanished_coords = (new_marginal <= cluster_mass_threshold)

        num_vanished_coords = np.sum(vanished_coords)
        # Beta has *not* crossed zero   ==>   *some* coordinate must be below the threshold.
        assert num_vanished_coords >= 1
        if num_vanished_coords > 1:
            debug_print("{} coordinates (> 1) vanished at bifurcation.".format(new_marginal))

        new_marginal[vanished_coords] = 0
        new_marginal = new_marginal / np.sum(new_marginal)

        beta = last_sol.beta
        debug_print("Reached bifurcation, {} solutions generated. Last solution before bifurcation: {}\n"
                    "Estimated marginal at bifurcation is {}; will use BA at "
                    "beta={:0.4g}".format(len(sols_between_bifs), last_sol, new_marginal, beta))

        # Line 9 at [1, Algorithm 3]: use BA to re-gain accuracy of reduced solution.
        # As before, dimensions are expanded back to problem's original dimensions.
        assert beta > 0
        last_sol = expand_back_sol_dims(BA_until_convergence(prob=reduced_prob, beta=beta,
                                                             initial_cluster_marginal=new_marginal,
                                                             compute_params=compute_params), clusters_with_mass)
        debug_print("Reduced solution computed with {} BA iterations:\n{}\n".format(last_sol.iter_count, last_sol))
        # The new `last_sol` is placed in the returned `solutions` list on the next `while` iteration, or below.

    if last_sol.beta > 0 and beta > 0:
        # Try not to return numeric solutions at negative beta values.
        assert support_size(last_sol) == 1

        # Off-grid points can be extrapolated later using the `extrapolation_funcs` property.
        # This was added at previous grid-points by _track_RD_root_to_bifurcation().
        # Allow the obvious (constant) extrapolation also from the trivial solution.
        polys = [np.poly1d(last_sol.p_t.flatten()[i]) for i in range(prob.T_dim)]
        last_sol.extrapolation_funcs = polys

        solutions.append(last_sol)

    debug_print("Done computing at beta={:0.4g}; support size={}".format(last_sol.beta, support_size(last_sol)))
    return list(reversed(solutions))


def extrapolate_intermediate_RTRD_solutions(RTRD_sols: [RD_defs.RD_sol_point],
                                            requested_betas) -> [RD_defs.RD_sol_point]:
    """
    Extrapolate off-grid solutions.

    Parameters:
            `RTRD_sols`         RD solutions, assumed to have been generated by RTRD_fixed_order_and_step_size().
            `requested_betas`   beta values at which to generated intermediate solutions.

    See also [1, Section I.3.2] on extrapolation.
    """
    assert all([hasattr(s, 'extrapolation_funcs') for s in RTRD_sols]), \
        "Input solutions must have an 'extrapolation_funcs' property, to allow extrapolation of intermediate " \
        "solutions. Were they generated using root-tracking for RD (RTRD_fixed_order_and_step_size)?"
    for i in range(len(RTRD_sols) - 1):
        assert RTRD_sols[i].beta < RTRD_sols[i + 1].beta, "Solutions assumed to be sorted by increasing beta value."
    # Start from the right.
    betas = sorted(requested_betas, reverse=True)

    extrapolated_sols = deque()
    for beta in betas:
        ind = 0
        # Use the nearest solution to the *right* of the requested beta.
        while ind + 1 < len(RTRD_sols) and RTRD_sols[ind].beta < beta:
            # TODO: Search using something faster such as binary search.
            ind += 1

        sol = RTRD_sols[ind]
        delta_beta = beta - sol.beta

        # Unless the extrapolated solution has a negative coordinate (can happen if there
        # is a bifurcation between consecutive solutions).
        extrapolated_marginal = np.array([p(delta_beta) for p in sol.extrapolation_funcs])
        if (extrapolated_marginal < 0).any():
            # Use solution to the *left* instead.
            if ind == 0:
                # No solution to the left.
                debug_print("Extrapolation at beta={:0.4g} is of negative marginal {}, "
                            "but no solution to the left -- stopping.".format(beta, extrapolated_marginal))
                break

            debug_print("Extrapolation at beta={:0.4g} is of negative marginal: {}; using solution to the "
                        "left instead.".format(beta, extrapolated_marginal))
            sol = RTRD_sols[ind - 1]
            delta_beta = beta - sol.beta
            extrapolated_marginal = np.array([p(delta_beta) for p in sol.extrapolation_funcs])
            assert (extrapolated_marginal >= 0).all()

        assert len(extrapolated_marginal) == len(sol.p_t)
        if np.abs(delta_beta) > 1:
            warnings.warn("Large deviation {:0.4g} from grid-point at {:0.4g}; extrapolation might "
                          "be imprecise.".format(delta_beta, sol.beta))

        # Ain't it nice, to have an approximate RD solution with *no* BA iterations used to compute it?   :-)
        extrapolated_sols.append(RD_defs.RD_sol_point(p_t=extrapolated_marginal, beta=beta))

    return list(reversed(extrapolated_sols))


def exact_solutions_for_binary_source_with_Hamming_distortion(p: float, requested_betas, deriv_order=0) \
        -> [RD_defs.RD_sol_point]:
    """
    Implements [1, Section III.F]: Exact solutions and their derivatives for a binary source
    with a Hamming distortion measure.

    If `deriv_order` then returned solutions have a 'derivatives' property,
    whose i-th entry contains the i-th order derivative with respect to beta.
    """
    assert deriv_order == int(deriv_order) >= 0
    assert 0. <= p < 0.5, "`p` < 0.5 is expected to represent the probability of a Bernoulli variable; got {}".format(p)
    betas = sorted(requested_betas)

    p_, beta = sympy.symbols('p beta')
    e = sympy.exp(beta)
    # Analytical solution by beta, [1, Eq. (F.5)]:
    probability_for_1_sympy = (1 - p_ * (1 + e)) / (1 - e)
    # Substitute the value `p`, leaving beta as sole variable.
    probability_for_1_evaluated = probability_for_1_sympy.subs({p_: p})
    # A numpy-compatible function for generating solutions as a function of beta.
    exact_solution = sympy.lambdify(beta,
                                    [1 - probability_for_1_evaluated, probability_for_1_evaluated],
                                    modules='numpy')

    def deriv_vector(order):
        """ A numpy-compatible analytical expression for solution's derivative with respect to beta. """
        deriv_sympy = sympy.factor(sympy.diff(probability_for_1_evaluated, beta, order))
        for_np = sympy.lambdify(beta, deriv_sympy, modules='numpy')
        return lambda b: np.array([-for_np(b), for_np(b)])

    exact_derivs = [deriv_vector(1 + d) for d in range(deriv_order)]

    # The critical beta value, [1, Eq. (F.6)]:
    beta_critical = np.log((1 - p) / p)
    # This is exactly where the probability for head vanishes:
    assert sympy.simplify(probability_for_1_sympy.subs({beta: sympy.log((1 - p_) / p_)})) == 0

    # Compute exact solutions.
    exact_sols = deque()
    for beta in betas:
        if beta < beta_critical:
            # The solution is trivial beyond the bifurcation point.
            exact_sols.append(RD_defs.RD_sol_point(p_t=np.array(exact_solution(beta_critical)), beta=beta))
        else:
            exact_sols.append(RD_defs.RD_sol_point(p_t=np.array(exact_solution(beta)), beta=beta))

    # Exact derivatives at exact solution, if requested.
    if deriv_order:
        for sol in exact_sols:
            if sol.beta < beta_critical:
                # Derivatives vanish at the constant solution-path.
                sol.derivatives = [np.zeros(2) for _ in range(deriv_order)]
            else:
                sol.derivatives = [exact_derivs[deg](sol.beta) for deg in range(deriv_order)]

    return exact_sols
