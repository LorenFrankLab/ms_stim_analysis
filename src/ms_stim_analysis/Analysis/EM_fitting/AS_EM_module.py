import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ms_stim_analysis.Analysis.EM_fitting.FiltersEM as ff

# Has just main functions which call others that do the heavy lifting

# State-space EM code in python
# Anne Smith 2015


################################
def RunEM(df, p_init=None, fig_ax_list=None, trial_number=None, color="b", label=None):
    startflag = 0
    sigma2e = 0.5**2  # start guess
    sigma_init = sigma2e

    x_init = 0.0

    if p_init is None:
        p_init = 0.5  # set default to .5

    if p_init > 0.01 and p_init < 0.99:
        mu = np.log(p_init / (1 - p_init))
    elif p_init <= 0.01:
        mu = -3.0
    else:
        mu = 3.0

    print("sigma2e:", sigma2e)
    x_post, sigma2_post, sigma2e, sigma_init, converge_flag = ff.EM(
        df.y, mu, sigma2e, x_init, sigma_init
    )

    pmode, p, pll, pul = ff.TransformToProb(x_post, sigma2_post, mu)

    if fig_ax_list is None:
        fig_ax_list = plt.subplots(figsize=figsize)
    fig, ax = fig_ax_list
    if trial_number is None:
        trial_number = np.arange(len(pmode))

    ccc = color
    trial_number_new = [trial_number[0]]
    pmode_new = [pmode[0]]
    pll_new = [pll[0]]
    pul_new = [pul[0]]
    for i in range(1, len(trial_number)):
        step = trial_number[i] - trial_number[i - 1]
        if step > 1:
            nan_array = np.nan * np.ones(step - 1)
            trial_number_new.extend(nan_array)
            pmode_new.extend(nan_array)
            pll_new.extend(nan_array)
            pul_new.extend(nan_array)
        trial_number_new.append(trial_number[i])
        pll_new.append(pll[i])
        pul_new.append(pul[i])
        pmode_new.append(pmode[i])

    ax.plot([0, len(p)], [0.5, 0.5], "k-", lw=1, alpha=0.5)
    ax.plot(
        trial_number_new,
        pmode_new,
        linestyle="-",
        color=ccc,
        alpha=0.9,
        lw=1.5,
        label=label,
    )
    # ax.plot(pll, linestyle = '-', color= ccc, alpha=0.9,lw=1.5)
    # ax.plot(pul, linestyle = '-', color= ccc, alpha=0.9,lw=1.5)
    ax.fill_between(trial_number_new, pll_new, pul_new, facecolor=ccc, alpha=0.3)
    ax.plot(range(1, len(df) + 1), df["y"], ".", color="gray", alpha=1, markersize=4)
    ax.locator_params(axis="y", nbins=3)

    # ax.plot([0, len(p)], [0.5, 0.5], "k-", lw=1, alpha=0.5)
    # ax.plot(
    #     trial_number, pmode, linestyle="-", color=ccc, alpha=0.9, lw=1.5, label=label
    # )
    # # ax.plot(pll, linestyle = '-', color= ccc, alpha=0.9,lw=1.5)
    # # ax.plot(pul, linestyle = '-', color= ccc, alpha=0.9,lw=1.5)
    # ax.fill_between(trial_number, pll, pul, facecolor=ccc, alpha=0.3)
    # ax.plot(range(1, len(df) + 1), df["y"], ".", color="gray", alpha=1, markersize=4)
    # ax.locator_params(axis="y", nbins=3)

    return fig, ax, pll, pul, pmode


###############################
def EM_main(resp_values, p_init=None, fig_ax_list=None, **kwargs):
    df = pd.DataFrame()
    df["y"] = resp_values
    fig, ax, pll, pul, pmode = RunEM(df, p_init, fig_ax_list, **kwargs)

    return fig, ax, pll, pul, pmode
