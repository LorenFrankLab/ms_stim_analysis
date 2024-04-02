import numpy as np
import matplotlib.pyplot as plt


def format_ax(ax,
              xlabel=None,
              ylabel=None,
              zlabel=None,
              title=None,
              title_color=None,
              xlim=None,
              ylim=None,
              xticks=None,
              xticklabels=None,
              yticks=None,
              yticklabels=None,
              fontsize=20,
              spines_off_list=["right", "top"]):
    """
    Format axis of plot.
    :param ax: axis object.
    :param xlabel: string. x label.
    :param ylabel: string. y label.
    :param title: string. Title.
    :param title_color: title color.
    :param xlim: list. x limits.
    :param ylim: list. y limits.
    :param xticks: list. x ticks.
    :param xticklabels: list. x tick labels.
    :param yticks: list. y ticks.
    :param yticklabels: list. y tick labels.
    :param fontsize: number. font size.
    :param spines_off_list: list. Remove these spines.
    :return:
    """
    # Define inputs if not passed
    if title_color is None:
        title_color = "black"
    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if zlabel is not None:
        ax.set_zlabel(zlabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize, color=title_color)
    # Ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    # Limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    for spine in spines_off_list:
        ax.spines[spine].set_visible(False)
    # Axis
    ax.tick_params(labelsize=fontsize)


def get_figsize(num_rows, num_columns, subplot_width, subplot_height):
    return (subplot_width*num_columns, subplot_height*num_rows)


def get_fig_axes(num_rows=None, num_columns=None, num_subplots=None, sharex=False, sharey=False, figsize=None,
                 subplot_width=None, subplot_height=None, remove_empty_subplots=True, gridspec_kw=None):
    # Check inputs not under or overspecified
    if np.sum([x is None for x in [num_rows, num_columns, num_subplots]]) != 1:
        raise Exception(f"Exactly two of num_rows, num_columns, num_subplots must be passed")
    if figsize is not None and (subplot_width is not None or subplot_height is not None):
        raise Exception(f"If figsize is passed, subplot_width and subplot_height should be None")

    # Define missing parameters
    if num_subplots is None:
        num_subplots = num_rows*num_columns
    if num_rows is None:
        num_rows = int(np.ceil(num_subplots/num_columns))
    elif num_columns is None:
        num_columns = int(np.ceil(num_subplots/num_rows))

    # If rows or columns is one, remove extra subplots if indicated
    if remove_empty_subplots and (num_rows == 1 or num_columns == 1) and num_rows * num_columns < num_subplots:
        if num_rows == 1:
            num_columns = num_subplots
        elif num_columns == 1:
            num_rows = num_subplots

    # Get figure size if not passed, using above params
    if figsize is None and subplot_width is not None and subplot_height is not None:
        figsize = get_figsize(num_rows, num_columns, subplot_width, subplot_height)

    # Return subplots
    return plt.subplots(num_rows, num_columns, sharex=sharex, sharey=sharey, figsize=figsize, gridspec_kw=gridspec_kw)
