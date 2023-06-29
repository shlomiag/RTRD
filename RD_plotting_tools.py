"""
Convenience functions for plotting the solutions of an RD problem.

See README.md for usage examples.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import log2, log10
from collections import deque
from collections.abc import Iterable
from pathlib import Path
import sys
import RD_defs
import root_tracking
import RD_root_tracking


_DEFAULT_ALPHA = 0.6
_ALPHA_BY_PROB_COUNT = lambda n: max(_DEFAULT_ALPHA * 0.8**(n - 1), 0.2)
_DEFAULT_MARKER = '1'
_MARKERS_LIST = [_DEFAULT_MARKER, '2', '3', '4', 'v', '^', '<', '>']
_DEFAULT_CLUSTER_MASS_THRESHOLD_FOR_PLOTTING_BIFS = 1.e-3

# Various constants used for plotting.
COLORS_LIST = ['b', 'g', 'c', 'm', 'r', 'k', 'y']

# Increase figure size, mainly for having saved plots reasonably sized.
FIG_SIZE_FACTOR = 2.5     # This used to be 2
_plot_fig_size = [int(FIG_SIZE_FACTOR*x) for x in matplotlib.rcParams["figure.figsize"]]
# Attempt to use external latex renderer.
matplotlib.rcParams['text.usetex'] = True


def __make_params_and_sols_to_lists(probs, compute_params, solutions):
    """
    Turn arguments into lists if they are not already so, supporting several use cases.

    Verifies that arguments are of type (RD_problem, RD_compute_params, list of RD_sol_point).
    """
    if isinstance(compute_params, Iterable):
        assert compute_params, "No computation parameters given!"
        _compute_params_list = compute_params
    else:
        _compute_params_list = (compute_params, )

    if isinstance(probs, Iterable):
        assert probs, "No problems given!"
        _probs_list = probs
    else:
        _probs_list = (probs, )

    assert solutions and isinstance(solutions, Iterable), "Expecting a non-empty list of solutions: nothing to plot."
    if isinstance(solutions[0], Iterable):
        _solutions_list = solutions
    else:
        _solutions_list = [solutions, ]

    # The number of distinct compute_params should match that the probs.
    # However, a single (prob, params) pair might be given with multiple
    # solution lists, e.g., when comparing different solution methods of
    # a problem.
    assert len(_probs_list) == len(_compute_params_list) > 0, \
        "Got {} many distinct problems, but {} many compute-params. " \
        "Numbers expected to match.".format(len(_solutions_list), len(_compute_params_list))

    if len(_probs_list) == len(_compute_params_list) == 1 < len(_solutions_list):
        _probs_list = _probs_list * len(_solutions_list)
        _compute_params_list = _compute_params_list * len(_solutions_list)
    else:
        assert len(_probs_list) == len(_compute_params_list) == len(_solutions_list), \
            "How should we match {} (prob, params) pairs to {} solution " \
            "lists?".format(len(_probs_list), len(_solutions_list))

    for prob, sols, params in zip(_probs_list, _solutions_list, _compute_params_list):
        assert isinstance(sols, Iterable)
        for s in sols:
            assert isinstance(s, RD_defs.RD_sol_point)

        assert isinstance(prob, RD_defs.RD_problem), "Expecting an RD_problem instance."
        assert isinstance(params, RD_defs.RD_compute_params), "Expecting an RD_compute_params instance."

    return _probs_list, _compute_params_list, _solutions_list


def _plot_setups(probs, compute_params, solutions, kwargs):
    """
    Setups before creating a plot. See also _plot_finish().

    Usage:
        _probs_list, _compute_params_list, _solutions_list = \
                _plot_setups(probs, compute_params, solutions, kwargs)

    Keyword args handled this function:         (partial)
        new_fig (True):             Create a new figure?
        labels (True):              List of per-problem labels to be used. Auto-generated if True.
        show_legend (see right):    Should legends be shown?
                                    Defaults to True if more than one solution list given, to False otherwise.

    Keyword args handled by _plot_finish():     (partial)
        save_to_filename (False):   Name of file (+extension) to which to save the figure.
        show_figure (True):         Should figure be shown?
        plot_title (False):         Plot title (string), or None.
        show_legend (None):         Should legend if True.
    """
    _probs_list, _compute_params_list, _solutions_list = \
        __make_params_and_sols_to_lists(probs=probs, compute_params=compute_params, solutions=solutions)

    save_to_filename = kwargs.get('save_to_filename')
    assert save_to_filename is None or save_to_filename is True or isinstance(save_to_filename, str), \
        "File name expected to be string if not None: %s." % str(save_to_filename)

    show_legend = kwargs.get('show_legend', None)
    if show_legend is None:
        kwargs['show_legend'] = True if len(_solutions_list) > 1 else None

    labels = kwargs.get('labels', True)
    if labels is True and (show_legend or len(_solutions_list) > 1):
        # kwargs dictionary passed by reference, so modified at caller's.
        kwargs['labels'] = [params.computation_method for params in _compute_params_list]

    if kwargs.get('new_fig', True):
        plt.figure(figsize=kwargs.get('fig_size') or _plot_fig_size)
    assert kwargs.get('new_figure') is None, "Did you mean 'new_fig=...'?"

    return _probs_list, _compute_params_list, _solutions_list


def __has_legit_suffix(filename):
    """ True if filename has an extension, False otherwise. """
    return Path(filename).suffix and len(Path(filename).suffix) < 4


def _plot_finish(kwargs, xlabel=None, ylabel=None):
    """
    A helper plot-finishes function, to be called when done plotting.
    """
    # Keyword arguments handled by this function:
    plot_title = kwargs.get('plot_title')
    save_to_filename = kwargs.get('save_to_filename')
    show_figure = kwargs.get('show_figure', True)
    show_legend = kwargs.get('show_legend')
    xlim, ylim = kwargs.get('xlim'), kwargs.get('ylim')
    xticks, yticks = kwargs.get('xticks'), kwargs.get('yticks')
    xlabel, ylabel = kwargs.get('xlabel', xlabel), kwargs.get('ylabel', ylabel)
    labelpad = kwargs.get('labelpad')
    label_fontsize = kwargs.get('fontsize') or kwargs.get('label_fontsize', 'x-large')
    legend_fontsize, legend_loc = kwargs.get('legend_fontsize', 'x-large'), kwargs.get('legend_loc')
    tick_params, text = kwargs.get('tick_params'), kwargs.get('text')
    dpi, grid = kwargs.get('dpi'), kwargs.get('grid')

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    if xticks:
        plt.xticks(xticks)

    if yticks:
        plt.yticks(yticks)

    if xlabel:
        plt.xlabel(xlabel, fontsize=label_fontsize, labelpad=labelpad)

    if ylabel:
        plt.ylabel(ylabel, fontsize=label_fontsize, labelpad=labelpad)

    if text:
        plt.text(**text)

    if tick_params:
        plt.tick_params(**tick_params)

    if show_legend:
        leg = plt.legend(fontsize=legend_fontsize, loc=legend_loc)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    if grid:
        plt.grid()

    if plot_title:
        plt.title(plot_title)

    if save_to_filename:
        if __has_legit_suffix(save_to_filename):
            plt.savefig(save_to_filename, dpi=dpi)
        else:
            plt.savefig(save_to_filename + '.pdf', dpi=dpi)
            plt.savefig(save_to_filename + '.png', dpi=dpi)
        print("Saved to file: " + save_to_filename)
        if not show_figure:
            plt.close()

    if show_figure:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()


def _find_bifurcations_in_sols(solutions,
                               cluster_mass_threshold=_DEFAULT_CLUSTER_MASS_THRESHOLD_FOR_PLOTTING_BIFS):
    """ Return bifurcations' indices in solutions. """
    clusters_in_support = [sol.p_t.flatten() > cluster_mass_threshold for sol in solutions]
    inds = deque()
    for i in range(len(solutions) - 1):
        if (clusters_in_support[i] != clusters_in_support[i + 1]).any():
            # Support has changed.
            already_in_list = False
            for j in inds:
                if np.abs(solutions[j].beta - solutions[i].beta) < .01:
                    already_in_list = True

            if not already_in_list:
                inds.append(i)

    return list(inds)


def _plot_bifurcations(solutions, plot_by_rate=False, prob=None, log2_beta_axis=True,
                       cluster_mass_threshold=_DEFAULT_CLUSTER_MASS_THRESHOLD_FOR_PLOTTING_BIFS):
    """ Add dashed-red verticals at bifurcation. """
    assert prob or not plot_by_rate, "Need problem definition to plot bifurcations by rate."

    bif_inds = _find_bifurcations_in_sols(solutions=solutions, cluster_mass_threshold=cluster_mass_threshold)
    my_log = log2 if log2_beta_axis else lambda x: x
    for i in bif_inds:
        value = solutions[i].rate(prob) if plot_by_rate else my_log(solutions[i].beta)
        plt.axvline(x=value, alpha=_DEFAULT_ALPHA, linestyle='--', c='r', linewidth=1.)


def plot_cluster_marginals_by_rate(probs, compute_params, solutions, **kwargs):
    """ Plot cluster marginal by rate. """
    kwargs['plot_by_rate'] = True
    plot_cluster_marginals(probs, compute_params, solutions, **kwargs)


def plot_cluster_marginals(probs, compute_params, solutions, **kwargs):
    """ Plot cluster marginals by beta. """
    _probs_list, _compute_params_list, _solutions_list = _plot_setups(probs, compute_params, solutions, kwargs)
    show_legend, labels = kwargs.get('show_legend'), kwargs.get('labels')
    log2_beta_axis, force_alpha = kwargs.get('log2_beta_axis', True), kwargs.get('alpha')
    plot_by_rate = kwargs.get('plot_by_rate')
    only_one_prob = len(_solutions_list) == 1
    marker_shape, linestyle = kwargs.get('marker'), kwargs.get('linestyle')
    plot_bifurcations = kwargs.get('plot_bifurcations', True)
    force_color, linewidth, markersize = kwargs.get('color'), kwargs.get('linewidth'), kwargs.get('markersize')
    color_i = kwargs.get('color_ind', kwargs.get('color_i', 0))
    if only_one_prob and kwargs.get('show_legend') is None:
        kwargs['show_legend'] = True
    if log2_beta_axis:
        beta_transform = lambda x: log2(x)
    else:
        beta_transform = lambda x: x

    for sol_list_i, (prob, params, sols) in enumerate(zip(_probs_list, _compute_params_list, _solutions_list)):
        horiz_axis = np.array([sol.rate(prob) for sol in sols]) if plot_by_rate \
            else np.array([beta_transform(sol.beta) for sol in sols])

        for t in range(prob.T_dim):
            marker = marker_shape if marker_shape else _MARKERS_LIST[(t + sol_list_i * prob.T_dim) % len(_MARKERS_LIST)]
            vals = np.array([sol.p_t[t] for sol in sols])

            if labels is not None and show_legend is not False:
                my_label = labels[sol_list_i] if t == 0 else None
            else:
                my_label = None if show_legend is False else r'$\hat{x}_' + str(1 + t) + r'$' if only_one_prob else None

            plt.plot(horiz_axis, vals, c=force_color or COLORS_LIST[t + color_i % len(COLORS_LIST)] if only_one_prob \
                else COLORS_LIST[sol_list_i + color_i % len(COLORS_LIST)],
                     alpha=force_alpha if force_alpha is not None else _ALPHA_BY_PROB_COUNT(len(_solutions_list)),
                     marker=marker, label=my_label, linestyle=linestyle, linewidth=linewidth, markersize=markersize)

        if plot_bifurcations:
            _plot_bifurcations(solutions=sols, plot_by_rate=plot_by_rate, prob=prob, log2_beta_axis=log2_beta_axis,
                               cluster_mass_threshold=kwargs.get('cluster_mass_threshold',
                                                                 _DEFAULT_CLUSTER_MASS_THRESHOLD_FOR_PLOTTING_BIFS))

    xlabel = r'$\textrm{Rate (bits per sample)}$' if plot_by_rate else \
        r'$\log_2 \beta$' if log2_beta_axis else r'$\beta$'
    _plot_finish(kwargs, xlabel=xlabel, ylabel=r'$p(\hat{x})$')


def plot_iter_count(probs, compute_params, solutions, **kwargs):
    """ Plot number of BA iterations used to compute each grid-point. """
    _, _compute_params_list, _solutions_list = _plot_setups(probs, compute_params, solutions, kwargs)
    show_legend, labels = kwargs.get('show_legend'), kwargs.get('labels')
    marker, color_ind = kwargs.get('marker'), kwargs.get('color_ind', 0)

    for sol_list_i, (params, sols) in enumerate(zip(_compute_params_list, _solutions_list)):
        log2_betas, iter_counts = deque(), deque()
        for sol in sols:
            if sol.iter_count is None:
                continue
            log2_betas.append(log2(sol.beta))
            iter_counts.append(sol.iter_count)

        plt.scatter(log2_betas, iter_counts, c=COLORS_LIST[sol_list_i + color_ind],
                    alpha=kwargs.get("alpha") or _ALPHA_BY_PROB_COUNT(len(_solutions_list)),
                    marker=marker or _MARKERS_LIST[sol_list_i], label=labels[sol_list_i] if show_legend else None,
                    s=kwargs.get("marker_size"))

        if kwargs.get("plot_bifurcations", True):
            _plot_bifurcations(solutions=sols)

    if plt.ylim()[1] > .5 * params.max_BA_iters_allowed and kwargs.get("plot_iter_limit", True):
        plt.axhline(y=params.max_BA_iters_allowed, c='r', ls='--', alpha=.9)
    plt.gca().set_yscale('log')

    _plot_finish(kwargs, xlabel=r'$\log_2 \beta$', ylabel=r'$\textrm{BA iterations}$')


def plot_eigenvals_Jacobian_wrt_cluster_marginals(probs, compute_params, solutions, **kwargs):
    """ Plot eigenvalues of BA operator in cluster-marginal coordinates. """
    _plot_Jacobian_eigenvals(probs=probs, compute_params=compute_params, solutions=solutions,
                             which_J_coords='cluster marginal', **kwargs)


def plot_eigenvals_Jacobian_wrt_direct_enc(probs, compute_params, solutions, **kwargs):
    """ Plot eigenvalues of BA operator in encoder coordinates. """
    _plot_Jacobian_eigenvals(probs=probs, compute_params=compute_params, solutions=solutions,
                             which_J_coords='direct encoder', **kwargs)


def _plot_Jacobian_eigenvals(probs, compute_params, solutions, which_J_coords, **kwargs):
    """
    Plot the Jacobian eigenvalues of Blahut-Arimoto's operator BA.

    Coordinates system is specified by `which_J_coords`: either 'cluster marginal' or 'direct encoder'.
    """
    ADMISSIBLE_COORDS = ('cluster marginal', 'direct encoder')
    assert which_J_coords in ADMISSIBLE_COORDS, 'Invalid choice of coordinates: "{}"'.format(which_J_coords)
    marginal_coords = (which_J_coords == 'cluster marginal')
    J_eigen_func_name = 'BA_Jacobian_eigenvals_wrt_marginal' if marginal_coords else \
        'BA_Jacobian_eigenvals_wrt_direct_enc'
    _probs_list, _compute_params_list, _solutions_list = _plot_setups(probs, compute_params, solutions, kwargs)
    show_legend, labels = kwargs.get('show_legend'), kwargs.get('labels')
    marker, linestyle = kwargs.get('marker'), kwargs.get('linestyle')

    for sol_list_i, (prob, params, sols) in enumerate(zip(_probs_list, _compute_params_list, _solutions_list)):
        log2_betas = np.array([log2(sol.beta) for sol in sols])
        eigenvals = 1 - np.array([sorted(getattr(sol, J_eigen_func_name)(prob)) for sol in sols]).T

        for t in range(eigenvals.shape[0]):
            plt.plot(log2_betas, eigenvals[t, :], c=COLORS_LIST[(t + prob.T_dim * sol_list_i) % len(COLORS_LIST)],
                     alpha=_ALPHA_BY_PROB_COUNT(len(_solutions_list)),
                     marker=marker or _MARKERS_LIST[(t + prob.T_dim * sol_list_i) % len(_MARKERS_LIST)],
                     label=labels[sol_list_i] if show_legend else None, linestyle=linestyle)

        _plot_bifurcations(solutions=sols)

    ylabel = r'$\textrm{eig } \nabla_{\log p(\hat{x})} \left( Id - BA_{\beta} \right)$' if marginal_coords \
        else r'$\textrm{eig } \nabla_{\log p(\hat{x}|x)} \left( Id - BA_{\beta} \right)$'
    _plot_finish(kwargs, xlabel=r'$\log_2 \beta$', ylabel=ylabel)


def plot_rate_distortion_curve(probs, compute_params, solutions, **kwargs):
    """ Plot the rate-distortion curve yielded by solutions. """
    _probs_list, _compute_params_list, _solutions_list = _plot_setups(probs, compute_params, solutions, kwargs)
    show_legend, labels, color_by_beta = kwargs.get('show_legend'), kwargs.get('labels'), kwargs.get('color_by_beta')
    force_color, force_alpha, linestyle = kwargs.get('c'), kwargs.get('alpha'), kwargs.get('linestyle')
    markersize = kwargs.get('markersize')
    if color_by_beta is None and len(_solutions_list) == 1:
        color_by_beta = True
    sol_list_start_i = kwargs.get('start_i', 0)

    cbar = None
    for sol_list_i, (prob, params, sols) in enumerate(zip(_probs_list, _compute_params_list, _solutions_list)):
        ind = sol_list_i + sol_list_start_i
        rates = np.array([sol.rate(prob) for sol in sols])
        distortions = np.array([sol.distortion(prob) for sol in sols])
        if color_by_beta:
            log2_beta = np.array([log2(sol.beta) for sol in sols])
        marker = kwargs.get('marker', _MARKERS_LIST[ind])

        color_by_beta = False
        # TODO: Coloring by beta currently works only with plt.scatter; can this be made to work also
        #       with plt.plot?
        scat = plt.plot(distortions, rates, c=force_color or (log2_beta if color_by_beta else COLORS_LIST[ind]),
                        alpha=force_alpha or _ALPHA_BY_PROB_COUNT(len(_solutions_list)), marker=marker,
                        label=labels[ind] if show_legend else None, linestyle=linestyle, markersize=markersize)

        if color_by_beta and cbar is None:
            cbar = plt.gcf().colorbar(scat)
            scat.set_clim([np.min(log2_beta), np.max(log2_beta)])

    if color_by_beta:
        cbar.set_label(r'$\log_{2} \beta$')
        # Avoid horizontal lines in colorbar
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")
        cbar.set_alpha(1)
        cbar.draw_all()

    _plot_finish(kwargs, xlabel=r'$\textrm{Distortion}$', ylabel=r'$\textrm{Rate (bits)}$')


def plot_error_from_reference_by_beta(probs, compute_params, reference_sols, compared_sols, log2_beta_axis=True,
                                      log10_error_val=True, beta0=None, prop_getter=lambda sol: sol.p_t,
                                      ylabel=r'$L_{\infty}\textrm{-error in } p(\hat{x})$', **kwargs):
    """
    Plot the L_infty-error of `compared_sols` from the `reference_sols`.

    The property used to measure error can be customized using `prop_getter`.
    Horizontal axis is shifted by `beta0` if not None.
    """
    assert isinstance(reference_sols, Iterable) and reference_sols
    for sol in reference_sols:
        assert isinstance(sol, RD_defs.RD_sol_point)

    _probs_list, _compute_params_list, _compared_sols_list = _plot_setups(probs, compute_params, compared_sols, kwargs)
    show_legend, labels = kwargs.get('show_legend'), kwargs.get('labels')
    force_marker, force_alpha, color = kwargs.get('marker'), kwargs.get('alpha'), kwargs.get('color')
    plot_title = kwargs.get('plot_title') or kwargs.get('title')
    if plot_title is None:
        kwargs['plot_title'] = 'Error for {} problem'.format(_probs_list[0].problem_name)

    reference_betas = np.array([sol.beta for sol in reference_sols])
    beta_vals_for_plotting = reference_betas - beta0 if beta0 is not None else reference_betas

    for sol_list_i, compared_sol_list in enumerate(_compared_sols_list):
        beta_vals_to_plot, vals_to_plot = deque(), deque()
        betas_of_compared_sols = [sol.beta for sol in compared_sol_list]

        for i, beta in enumerate(reference_betas):
            try:
                ind_in_compared_list = betas_of_compared_sols.index(beta)
            except ValueError:
                # Exact value we're searching for is not on the list.
                continue

            sol, compared_sol = reference_sols[i], compared_sol_list[ind_in_compared_list]
            assert sol.beta == compared_sol.beta

            err_val = np.max(np.abs((prop_getter(sol) - prop_getter(compared_sol))))
            vals_to_plot.append(log10(err_val) if log10_error_val else err_val)
            b = log2(beta_vals_for_plotting[i]) if log2_beta_axis else beta_vals_for_plotting[i]
            beta_vals_to_plot.append(b)

        plt.scatter(x=beta_vals_to_plot, y=vals_to_plot, c=color or COLORS_LIST[sol_list_i % len(COLORS_LIST)],
                    alpha=force_alpha or _ALPHA_BY_PROB_COUNT(len(_compared_sols_list)),
                    marker=force_marker or _MARKERS_LIST[sol_list_i % len(_MARKERS_LIST)],
                    label=labels[sol_list_i] if show_legend else None, s=kwargs.get('marker_size'))

    _plot_bifurcations(solutions=reference_sols, log2_beta_axis=log2_beta_axis)
    dbeta_label = r'(\beta - \beta_0)' if beta0 is not None else r'\beta'
    _ylabel = r'$\log_{10} \textrm{ of }' + ylabel[1:] if log10_error_val else ylabel
    _plot_finish(kwargs, xlabel=r'$\log_2 ' + dbeta_label + r'$' if log2_beta_axis else r'$' + dbeta_label + r'$',
                 ylabel=_ylabel)


def plot_derivatives_norm_by_beta(probs, compute_params, solutions, max_deriv_order, **kwargs):
    """ Plot the L2-norm of the implicit numerical derivatives, of orders 1...`max_deriv_order`. """
    if kwargs.get('show_legend') is None:
        kwargs['show_legend'] = True
    _probs_list, _compute_params_list, _solutions_list = _plot_setups(probs, compute_params, solutions, kwargs)
    if kwargs.get('plot_title') is None:
        kwargs['plot_title'] = "Derivatives' norms, for " + _probs_list[0].problem_name + " problem"

    assert len(_probs_list) == len(_compute_params_list) == len(_solutions_list) == 1
    prob, params, sols = _probs_list[0], _compute_params_list[0], _solutions_list[0]

    betas, vals = deque(), deque()
    # Collect numerical implicit derivatives. Computing by grid-point is more efficient than by derivative's order.
    for i, sol in enumerate(reversed(sols)):
        if i % 100 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

        try:
            deriv_tensor_calc = \
                RD_root_tracking.IdMinusBADerivativeTensorsInMarginalCoordinates(solution_point=sol,
                                                                                 problem_def=prob,
                                                                                 max_deriv_order=max_deriv_order,
                                                                                 _debug=False)
            tracker = root_tracking.RootTracker(deriv_tensor_calc=deriv_tensor_calc, _debug=False)
            vals.append([log10(np.sqrt(np.sum(tracker.calculate_implicit_derivative(deriv_order=1 + i) ** 2)))
                         for i in range(max_deriv_order)])
            betas.append(sol.beta)
            deriv_tensor_calc.clear_caches()

        except AssertionError:
            # Solution might be too close to bifurcation to compute, etc. Ignore these.
            break

    for i in range(max_deriv_order):
        plt.plot(x=betas, y=[v[i] for v in vals], c=COLORS_LIST[i % len(COLORS_LIST)],
                 alpha=_ALPHA_BY_PROB_COUNT(max_deriv_order),
                 marker=_MARKERS_LIST[i % len(_MARKERS_LIST)],
                 label=r'$\#' + str(i + 1) + r'\textrm{ derivative}$')

    _plot_bifurcations(solutions=sols, log2_beta_axis=False)

    _plot_finish(kwargs, xlabel=r'$\beta$', ylabel=r"$\log_{10}(L_2 \textrm{ of derivatives' norm})$")

