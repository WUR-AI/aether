##############################
## GENERAL PLOTTING FUNCTIONS
## Copied from
##############################

import copy

import matplotlib.pyplot as plt
import numpy as np


def dummy_example_functions(fig=None, ax=None):
    """Example of common input arguments."""
    if ax is None:
        if fig is None:
            ## Create both a figure & axis handle
            fig, ax = plt.subplot(111)
        else:
            ## Create single axis on figure
            ax = fig.add_subplot(111)
    else:
        ## use provided axis, figure must already exist so don't create new one:
        pass

    return fig, ax


def despine(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def naked(ax):
    """Remove all spines, ticks and labels."""
    for ax_name in ["top", "bottom", "right", "left"]:
        ax.spines[ax_name].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def set_fontsize(font_size=12):
    """Set font size for all text elements."""
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.autolimit_mode"] = "data"  # default: 'data'
    params = {
        "legend.fontsize": font_size,  # set for all kinds of text elements
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    }
    plt.rcParams.update(params)
    print(f"Font size is set to {font_size}")


def get_list_ticks(ax, axis="x"):
    assert axis in ["x", "y"]
    if axis == "x":
        return ax.get_xticks().tolist()
    elif axis == "y":
        return ax.get_yticks().tolist()


def get_list_ticklabels(ax, axis="x"):
    """Note: only works if figure has been drawn. Otherwise returns empty list.
    To draw figure, use plt.show() or plt.draw()"""
    assert axis in ["x", "y"]
    if axis == "x":
        return [x.get_text() for x in ax.get_xticklabels()]
    elif axis == "y":
        return [x.get_text() for x in ax.get_yticklabels()]


def set_min_lims_to_0(ax, set_x=True, set_y=True):
    """Set minimum limits to 0."""
    if set_x:
        ax.set_xlim(left=0)
    if set_y:
        ax.set_ylim(bottom=0)


def equal_xy_lims(ax, start_zero=False):
    """Set xlim equal to ylim (by their max)"""
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    max_outer_lim = np.maximum(xlims[1], ylims[1])
    min_inner_lim = np.minimum(xlims[0], ylims[0])

    if start_zero:
        ax.set_xlim([0, max_outer_lim])
        ax.set_ylim([0, max_outer_lim])
    else:
        ax.set_xlim([min_inner_lim, max_outer_lim])
        ax.set_ylim([min_inner_lim, max_outer_lim])


def equal_lims_two_axs(ax1, ax2):
    """Set xlim equal to ylim across two axes."""
    xlim_1 = ax1.get_xlim()
    xlim_2 = ax2.get_xlim()
    ylim_1 = ax1.get_ylim()
    ylim_2 = ax2.get_ylim()

    new_x_min = np.minimum(xlim_1[0], xlim_2[0])
    new_x_max = np.maximum(xlim_1[1], xlim_2[1])
    new_y_min = np.minimum(ylim_1[0], ylim_2[0])
    new_y_max = np.maximum(ylim_1[1], ylim_2[1])

    ax1.set_xlim([new_x_min, new_x_max])
    ax2.set_xlim([new_x_min, new_x_max])
    ax1.set_ylim([new_y_min, new_y_max])
    ax2.set_ylim([new_y_min, new_y_max])


def equal_lims_n_axs(ax_list):
    """Set ax lims equal across list of ax, but xlim and ylim are separate here."""
    for i_ax, ax in enumerate(ax_list):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if i_ax == 0:
            x_min_min = copy.deepcopy(x_min)
            x_max_max = copy.deepcopy(x_max)
            y_min_min = copy.deepcopy(y_min)
            y_max_max = copy.deepcopy(y_max)

        else:
            if x_min < x_min_min:
                x_min_min = copy.deepcopy(x_min)
            if x_max > x_max_max:
                x_max_max = copy.deepcopy(x_max)
            if y_min < y_min_min:
                y_min_min = copy.deepcopy(y_min)
            if y_max > y_max_max:
                y_max_max = copy.deepcopy(y_max)

    for ax in ax_list:
        ax.set_xlim([x_min_min, x_max_max])
        ax.set_ylim([y_min_min, y_max_max])


def remove_xticklabels(ax):
    """Remove x ticklabels but keep ticks."""
    ax.set_xticklabels(["" for x in ax.get_xticklabels()])


def remove_yticklabels(ax):
    """Remove y ticklabels but keep ticks."""
    ax.set_yticklabels(["" for x in ax.get_yticklabels()])


def remove_both_ticklabels(ax):
    """Both x and y."""
    remove_xticklabels(ax)
    remove_yticklabels(ax)


def add_panel_label(
    ax,
    fig=None,
    label_letter="A",
    label_ind=None,
    uppercase=False,
    weight="bold",
    fontsize=None,
    verbose=0,
    method_placement="labels",
    x_override=None,
    y_override=None,
    x_offset=0,
    y_offset=0,
):
    """Add a panel label to ax. Either the str given by label_letter or the i-th letter given by
    label_ind. Panel label is automatically placed at top-left corner. But can be overridden by
    using x_override and y_override. By default using regular fontsize, but can be overridden by
    setting fontsize.

    IMPORTANT: run this function AFTER figure is drawn (plt.draw())

    Coordinates are in axes fraction.
    """
    ## Define label: either label_letter or label_ind
    if label_letter is None:
        assert isinstance(
            label_ind, int
        ), f"label_ind should be an int, but is a {type(label_ind)}"
        letters = "abcdefghijklmnopqrstuvwxyz"
        label_letter = letters[label_ind]
        if uppercase:
            label_letter = label_letter.upper()
    else:
        assert isinstance(
            label_letter, str
        ), f"label_letter should be a str, but is a {type(label_letter)}"
        if label_ind is not None:
            print("WARNING: both label_letter and label_ind are not None. Using label_letter.")

    ## Find coordinates:
    assert method_placement in [
        "labels",
        "blind",
        "override",
    ], f"method_placement={method_placement} not recognized"

    if method_placement == "labels":
        str_ylabel = ax.get_ylabel()
        str_xlabel = ax.get_xlabel()
        str_title = ax.get_title()
        bool_ylabel = str_ylabel != ""
        bool_xlabel = str_xlabel != ""
        bool_title = str_title != ""

        xcoord_ylabel = ax.yaxis.label.get_position()[0]  # get x coord of ylabel IN FIG PIXELS
        tmp = ax.transAxes.inverted().transform((xcoord_ylabel, 0))  # convert to axes fraction
        xcoord_ylabel = tmp[0]  # get x coord of ylabel IN AXES fraction
        if bool_ylabel:  # if there is a ylabel, use that as left-most x coord
            xcoord_left = xcoord_ylabel
        else:  # if no ylabel, move panel label inwards
            xcoord_left = np.minimum(xcoord_ylabel + 0.05, 0)
        if bool_title:  # if there is a title, use that as top-most y coord
            ycoord_title = ax.title.get_position()[1]  # get y coord of title IN AX fraction
            ycoord_top = ycoord_title
        else:  # if no title, use align with top of fig
            ycoord_top = 1.0

        ## Offset if necessary:
        if x_offset is not None:
            xcoord_left += x_offset
        if y_offset is not None:
            ycoord_top += y_offset

    elif method_placement == "blind":
        xcoord_left = -0.1
        ycoord_top = 1.05

    elif method_placement == "override":
        assert x_override is not None, "x_override must be specified"
        assert y_override is not None, "y_override must be specified"
        xcoord_left = x_override
        ycoord_top = y_override

    ## Override if necessary
    if x_override is not None:
        xcoord_left = x_override
    if y_override is not None:
        ycoord_top = y_override

    ## Create label, if verbose print coords.
    if verbose > 0:
        print(f"Coordinates: x {xcoord_left}, y {ycoord_top}")
    if fontsize is None:
        fontsize = plt.rcParams["font.size"]

    ax.annotate(
        label_letter,
        xy=(xcoord_left, ycoord_top),
        # xycoords='figure pixels',
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        weight=weight,
        fontsize=fontsize,
    )


def unpack_2d_list(list2d):
    """Unpack a 2d list into a 1d list."""
    return [item for sublist in list2d for item in sublist]


def concat_all_dict_values_into_list(dict_of_lists):
    """Concat all values in dict into a single list."""
    return [item for sublist in dict_of_lists.values() for item in sublist.values()]


def two_digit_sci_not(x):
    """Convert a float to a string in scientific notation with 2 digits in the exponent and no
    decimals."""
    sci_not_spars = np.format_float_scientific(x, precision=1)
    ## If precision=1, then it's either ceil or exactly precision=0
    if sci_not_spars[2] == "e":  # exactly precision=0 so one shorter
        sci_not_spars = sci_not_spars[0] + sci_not_spars[2:]  # skip dot
    elif sci_not_spars[3] == "e":  # ceil
        sci_not_spars = str(int(sci_not_spars[0]) + 1) + sci_not_spars[3:]  # skip dot
    else:
        assert (
            False
        ), f"precision=1 but neither ceil nor exactly precision=0. sci_not_spars={sci_not_spars}"
    return sci_not_spars


def readable_p(p_val):
    """Convert a scientific-notation p-value (str) to a readable string (to be formatted by
    matplotlib).

    Always ceil the decimal (for rounding, see readable_p_exact()). If a float is given, it is
    first converted to a string using two_digit_sci_not().
    """

    if not isinstance(p_val, str):
        p_val = two_digit_sci_not(x=p_val)
    if p_val[2] == "e":
        assert p_val[:4] == "10e-", p_val
        tmp_exp = int(p_val[-2:])
        p_val = f"1e-{str(tmp_exp - 1).zfill(2)}"

    if len(p_val) > 5:
        assert len(p_val) == 6 and p_val[1:3] == "e-", p_val
        exponent = p_val[-3:]
        read_p = f"{p_val[0]}x" + r"$10^{{-{tmp}}}$".format(
            tmp=exponent
        )  # for curly brackets explanation see https://stackoverflow.com/questions/53781815/superscript-format-in-matplotlib-plot-legend
    else:
        if p_val == "1e+00" or p_val == "1e-00":
            read_p = "1.0"
        else:
            assert p_val[2] == "-", f"p value is greater than 1, p val: {p_val}"

            if p_val[-3:] == "-01":
                read_p = f"0.{p_val[0]}"
            elif p_val[-3:] == "-02":
                read_p = f"0.0{p_val[0]}"
            elif p_val[-3:] == "-03":
                read_p = f"0.00{p_val[0]}"
            else:
                if int(p_val[-2:]) < 10:
                    exponent = p_val[-1]
                else:
                    exponent = p_val[-2:]
                read_p = f"{p_val[0]}x" + r"$10^{{-{tmp}}}$".format(
                    tmp=exponent
                )  # for curly brackets explanation see https://stackoverflow.com/questions/53781815/superscript-format-in-matplotlib-plot-legend
    return read_p


def readable_p_exact(p_val):
    """Convert a p-value (float) to a readable string, without ceiling."""
    if not isinstance(p_val, float):
        p_val = float(p_val)

    if p_val >= 0.01:
        read_p = str(np.round(p_val, 2))
    elif p_val >= 0.001:
        read_p = str(np.round(p_val, 3))
    else:
        tmp = np.format_float_scientific(p_val, precision=0)  # round to nearest
        p_val = tmp[0] + tmp[2:]  # skip dot
        if p_val[2] == "e":
            assert p_val[:4] == "10e-", p_val
            tmp_exp = int(p_val[-2:])
            p_val = f"1e-{str(tmp_exp - 1).zfill(2)}"

        if len(p_val) > 5:
            assert len(p_val) == 6 and p_val[1:3] == "e-", p_val
            exponent = p_val[-3:]
            read_p = f"{p_val[0]}x" + r"$10^{{-{tmp}}}$".format(
                tmp=exponent
            )  # for curly brackets explanation see https://stackoverflow.com/questions/53781815/superscript-format-in-matplotlib-plot-legend
        else:
            assert p_val[2] == "-", f"p value is greater than 1, p val: {p_val}"

            if int(p_val[-2:]) < 10:
                exponent = p_val[-1]
            else:
                exponent = p_val[-2:]
            read_p = f"{p_val[0]}x" + r"$10^{{-{tmp}}}$".format(
                tmp=exponent
            )  # for curly brackets explanation see https://stackoverflow.com/questions/53781815/superscript-format-in-matplotlib-plot-legend

            if read_p == "1x$10^{-3}$":  # can happen from rounding up fronm eg 0.0099
                read_p = "0.001"
    return read_p


def readable_p_significance_statement(
    p_val,
    n_bonf=None,
    upper_criterion_1_asterisk=0.05,
    upper_criterion_2_asterisks=0.01,
    upper_criterion_3_asterisks=0.001,
):
    """Convert a p-value (float) to significance notation (p < 0.001, p < 0.01, p < 0.05, n.s.; as
    well as asterisks)."""
    assert isinstance(upper_criterion_1_asterisk, float)
    assert isinstance(upper_criterion_2_asterisks, float)
    assert isinstance(upper_criterion_3_asterisks, float)

    if not isinstance(p_val, float):
        p_val = float(p_val)
    if n_bonf is None:
        n_bonf = 1

    if p_val < upper_criterion_3_asterisks / n_bonf:
        str_p = f"p < {upper_criterion_3_asterisks}"
        str_short = "***"
    elif p_val < upper_criterion_2_asterisks / n_bonf:
        str_p = f"p < {upper_criterion_2_asterisks}"
        str_short = "**"
    elif p_val < upper_criterion_1_asterisk / n_bonf:
        str_p = f"p < {upper_criterion_1_asterisk}"
        str_short = "*"
    else:
        str_p = f"p = {np.round(p_val, 2)}"
        str_short = "n.s."

    return str_p, str_short
