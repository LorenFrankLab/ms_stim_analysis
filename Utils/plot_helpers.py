import numpy as np
import matplotlib.pyplot as plt


def format_ax(
    ax,
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
    spines_off_list=["right", "top"],
):
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
