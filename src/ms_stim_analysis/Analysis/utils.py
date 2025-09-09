from spyglass.common import Session, interval_list_intersect
from datajoint.user_tables import UserTable
import numpy as np
from typing import Tuple
from tqdm import tqdm

import scipy
import matplotlib.pyplot as plt
from ms_stim_analysis.AnalysisTables.ms_task_identification import TaskIdentification
from ms_stim_analysis.AnalysisTables.ms_interval import EpochIntervalListName

from ms_stim_analysis.AnalysisTables.ms_opto_stim_protocol import (
    OptoStimProtocol,
    OptoStimProtocolTransfected,
    OptoStimProtocolLaser,
    OptoStimProtocolClosedLoop,
)
from .position_analysis import get_running_intervals, filter_position_ports


def filter_animal(table: UserTable, animal: str) -> UserTable:
    """filter table with all sessions for an animal

    Parameters
    ----------
    table : UserTable
        table to filter
    animal : str
        animal to include

    Returns
    -------
    UserTable
        filtered table
    """
    if len(animal) == 0:
        return table
    animal_key = {"subject_id": animal}
    if animal == "transfected":
        animal_key = [
            {"subject_id": name}
            for name in ["Totoro", "Winnie", "Banner", "Frodo", "Odins"]
        ]
    elif animal == "control":
        animal_key = [
            {"subject_id": name} for name in ["Yoshi", "Olive", "Wallie", "Bilbo"]
        ]
    return table & ((table * Session) & animal_key).fetch("KEY")


def filter_task(table: UserTable, task: str) -> UserTable:
    """filter table with all epochs for a given task type (e.g. "lineartrack)

    Parameters
    ----------
    table : UserTable
        table to filter
    task : str
        task type to include

    Returns
    -------
    UserTable
        filtered table
    """
    if task == "early_wtrack":
        filter_table = filter_task(table, "wtrack")
        return filter_table & early_wtrack_keys

    if task == "first_wtrack":
        filter_table = filter_task(table, "wtrack")
        return filter_table & first_wtrack_keys

    wtrack_aliases = ["wtrack", "w-track", "w track", "W-track", "W track", "Wtrack"]
    lineartrack_aliases = [
        "lineartrack",
        "linear-track",
        "linear track",
        "Linear-track",
        "Linear track",
        "Lineartrack",
    ]
    alias_sets = [wtrack_aliases, lineartrack_aliases]

    if len(task) == 0:
        return table
    for alias_set in alias_sets:
        if task in alias_set:
            keys = []
            for alias in alias_set:
                keys.extend(
                    (
                        table * EpochIntervalListName * TaskIdentification
                        & {"contingency": alias}
                    ).fetch("KEY")
                )
            return table & keys

    return table & (
        table * EpochIntervalListName * TaskIdentification & {"contingency": task}
    ).fetch("KEY")


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def convert_delta_marks_to_timestamp_values(
    marks: list, mark_timestamps: list, sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """convert delta marks to data values at regularly sampled timestamps

    Parameters
    ----------
    marks : list
        values of delta marks
    mark_timestamps : list
        when the marks occur
    sampling_rate : int
        desired sampling rate

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        data, timestamps
    """
    timestamps = np.arange(
        mark_timestamps[0],
        mark_timestamps[-1],
        1 / sampling_rate,
    )
    data = np.zeros(len(timestamps))

    ind = np.argmin(np.abs(timestamps - mark_timestamps[0]))
    data[:ind] = 1 - marks[0]
    for i in range(len(marks) - 1):
        ind_new = np.argmin(np.abs(timestamps - mark_timestamps[i]))
        data[ind:ind_new] = 1 - marks[i + 1]
        ind = ind_new.copy()
    return data, timestamps


def bootstrap(x, n_bootstraps, func=np.mean):
    """Bootstrap a function func by sampling with replacement from x."""
    n = len(x)
    idx = np.random.randint(0, n, (n_bootstraps, n))
    return func(np.array(x)[idx], axis=1)


def bootstrap_general(
    data,
    sample_size=None,
    statistic=np.median,
    conf_interval=95,
    n_boot=1e5,
    return_samples=False,
    **kwargs,
):
    if sample_size is None:
        sample_size = data.shape[0]
    bootstrap = []
    for i in range(int(n_boot)):
        #             bootstrap.append(statistic(np.random.choice(data,sample_size)))
        bootstrap.append(
            statistic(
                data[np.random.choice(np.arange(data.shape[0]), sample_size)], **kwargs
            )
        )
    if return_samples:
        return (
            np.mean(bootstrap, axis=0),
            [
                np.percentile(bootstrap, (100 - conf_interval) / 2, axis=0),
                np.percentile(
                    bootstrap, conf_interval + (100 - conf_interval) / 2, axis=0
                ),
            ],
            bootstrap,
        )

    return np.mean(bootstrap, axis=0), [
        np.percentile(bootstrap, (100 - conf_interval) / 2, axis=0),
        np.percentile(bootstrap, conf_interval + (100 - conf_interval) / 2, axis=0),
    ]


def bootstrap_diff(
    data1,
    data2,
    measurement=None,
    sample_size=None,
    statistic=np.mean,
    n_boot=1e3,
    conf_interval=95,
    return_samples=False,
    **kwargs,
):
    """bootstrap comparison of samples from 2 datasets

    Args:
        data1: first dataset
        data2: second dataset
        measurement: the value being calculated from samples (see measurement.py)
        sample_size (_type_, optional): _description_. Defaults to None.
        statistic (_type_, optional): how to compile across trials. Defaults to np.mean.
        n_boot: number of bootstrap samples. Defaults to 1e3.
        conf_interval (int, optional): confidence interval bounds to return. Defaults to 95.
        return_samples (bool, optional): whether to return the bootstrapped distribution. Defaults to False.

    Returns:
        _type_: _description_
    """
    y, rng, boot = bootstrap_compare(
        data1,
        data2,
        np.subtract,
        measurement,
        sample_size,
        statistic,
        n_boot,
        conf_interval,
        return_samples=True,
        **kwargs,
    )
    if return_samples:
        if rng[0] > 0 or rng[1] < 0:
            return y, rng, True, boot
        else:
            return y, rng, False, boot
    if rng[0] > 0 or rng[1] < 0:
        return y, rng, True
    else:
        return y, rng, False


def bootstrap_compare(
    data1,
    data2,
    operator=np.subtract,
    measurement=None,
    sample_size=None,
    statistic=np.mean,
    n_boot=1e3,
    conf_interval=95,
    return_samples=False,
    **kwargs,
):
    """bootstap comparison of samples from 2 datasets"""
    """bootstrap comparison of samples from 2 datasets
    data1: first dataset
    data2: second dataset
    operator: how we compare the measurements
    measurement: the value being calculated from samples (see measurement.py)
    ***for efficiency, precalculate measurement and use None value for non-population averaged measures
    statistic: what value of the distribution of operator results we care about (*irrelevant for population based measures)
    n_boot: number of bootstrap samples
    conf_interval: confidence interval
    return_samples: whether to return the samples

    Returns:
    mean: mean of the bootstrap samples
    confidence interval
    whether the confidence interval does not include 0
    """
    # measurement: the value being calculated from samples (see measurement.py)
    #    ***for efficiency, precalculate measurement and use None value for non-population averaged measures
    # operator: how we compare the measurements
    # statistic: what value of the distribution of operator results we care about (*irrelevant for population based measures)

    if sample_size is None:
        sample_size = data1.shape[0]  # min(data1.shape[0],data2.shape[0])
    if measurement is None:
        measurement = lambda x: x
    bootstrap = []
    for i in tqdm(range(int(n_boot)), position=0, leave=True):
        #     for i in range(int(n_boot)):
        bootstrap.append(
            statistic(
                operator(
                    measurement(
                        data1[np.random.choice(np.arange(data1.shape[0]), sample_size)],
                        **kwargs,
                    ),
                    measurement(
                        data2[np.random.choice(np.arange(data2.shape[0]), sample_size)],
                        **kwargs,
                    ),
                )
            )
        )
    bootstrap = np.array(bootstrap)
    if return_samples:
        return (
            np.mean(bootstrap, axis=0),
            [
                np.percentile(bootstrap, (100 - conf_interval) / 2, axis=0),
                np.percentile(
                    bootstrap, conf_interval + (100 - conf_interval) / 2, axis=0
                ),
            ],
            bootstrap,
        )

    return np.mean(bootstrap, axis=0), [
        np.percentile(bootstrap, (100 - conf_interval) / 2, axis=0),
        np.percentile(bootstrap, conf_interval + (100 - conf_interval) / 2, axis=0),
    ]


def bootstrap_traces(
    data,
    sample_size=None,
    statistic=np.mean,
    n_boot=1e3,
    conf_interval=95,
):
    if sample_size is None:
        sample_size = data.shape[0]
    bootstrap = []
    #     for i in tqdm(range(int(n_boot)),position=0,leave=True):
    for i in range(int(n_boot)):
        bootstrap.append(
            statistic(
                data[np.random.choice(np.arange(data.shape[0]), sample_size), :], axis=0
            )
        )
    bootstrap = np.array(bootstrap)
    return np.mean(bootstrap, axis=0), [
        np.percentile(bootstrap, (100 - conf_interval) / 2, axis=0),
        np.percentile(bootstrap, conf_interval + (100 - conf_interval) / 2, axis=0),
    ]


def get_running_valid_intervals(
    pos_key: dict,
    filter_speed: float = 10,
    filter_ports: bool = True,
    seperate_optogenetics: bool = True,
    dlc_pos: bool = False,
):
    """Find intervals where rat is running and not in a port.  if seperate_optogenetics, then also separate into intervals where optogenetics are and ar not running

    Args:
        pos_key (dict): key to find the position data
        filter_speed (float, optional): speed threshold for running. Defaults to 10.
        filter_ports (bool, optional): whether to filter out port intervals. Defaults to True.
        seperate_optogenetics (bool, optional): whether to seperate into optogenetic and control intervals. Defaults to True.
        dlc_pos (bool, optional): whether to use DLC position data. Defaults to False.

    Returns:
        if not seperate_optogenetics:
        run_intervals (list): intervals where rat is running
        if seperate_optogenetics:
        optogenetic_run_interval (list): intervals where rat is running and in optogenetic interval
        control_run_interval (list): intervals where rat is running and in control interval
    """
    # make intervals where rat is running
    run_intervals = get_running_intervals(
        **pos_key, filter_speed=filter_speed, dlc_pos=dlc_pos
    )
    # intersect with position-defined intervals
    if filter_ports:
        valid_position_intervals = filter_position_ports(pos_key, dlc_pos=dlc_pos)
        run_intervals = interval_list_intersect(
            np.array(run_intervals), np.array(valid_position_intervals)
        )
    if not seperate_optogenetics:
        return run_intervals

    # determine if each interval is in the optogenetic control interval
    control_interval = (OptoStimProtocol() & pos_key).fetch1("control_intervals")
    test_interval = (OptoStimProtocol() & pos_key).fetch1("test_intervals")
    if len(control_interval) == 0 or len(test_interval) == 0:
        print(f"Warning: no optogenetic intervals found for {pos_key}")
        return np.array([]), np.array([])
    optogenetic_run_interval = interval_list_intersect(
        np.array(run_intervals), np.array(test_interval)
    )
    control_run_interval = interval_list_intersect(
        np.array(run_intervals), np.array(control_interval)
    )
    return optogenetic_run_interval, control_run_interval


def autocorr2d(x):
    """Efficiently compute autocorrelation along 1 axis of a 2D array

    Args:
        x (np.array): data to compute autocorrelation of (n_samples, n_features)

    Returns:
        corr (np.array): autocorrelation of x along axis 0 (n_samples, n_features)
    """
    n = x.shape[0]
    # Zero-pad the array for FFT-based convolution
    padded_x = np.pad(x, ((0, n), (0, 0)), "constant")

    # Compute FFT and its complex conjugate
    X_f = np.fft.fft(padded_x, axis=0)
    result = np.fft.ifft(X_f * np.conj(X_f), axis=0).real

    # Return the positive lags
    return result[:n] / result[0]


def filter_opto_data(dataset_key: dict):
    """filter optogenetic data based on the dataset key

    Args:
        dataset_key (dict): restriction to filter by

    Returns:
        Table: filtered table
    """
    # define datasets
    dataset_table = OptoStimProtocol
    if "transfected" in dataset_key:
        dataset_table = dataset_table * OptoStimProtocolTransfected
    if "laser_power" in dataset_key:
        dataset_table = dataset_table * OptoStimProtocolLaser
    if "targeted_phase" in dataset_key:
        dataset_table = dataset_table * OptoStimProtocolClosedLoop
    dataset = dataset_table & dataset_key
    if "animal" in dataset_key:
        dataset = filter_animal(dataset, dataset_key["animal"])
    if "track_type" in dataset_key:
        dataset = filter_task(dataset, dataset_key["track_type"])
    if "min_pulse_length" in dataset_key:
        dataset = dataset & f"pulse_length_ms>{dataset_key['min_pulse_length']}"
    if "max_pulse_length" in dataset_key:
        dataset = dataset & f"pulse_length_ms<{dataset_key['max_pulse_length']}"
    print("datasets:", len(dataset))
    return dataset


def smooth(data, n=5, sigma=None, hamming=False):
    """smooths data with gaussian kernel of size n"""
    if n % 2 == 0:
        n += 1  # make sure n is odd
    if sigma is None:
        sigma = n / 2
    kernel = gkern(n, sigma)[:, None]
    if hamming:
        kernel = np.ones((n, 1)) / n
    if len(data.shape) == 1:
        pad = np.ones(((n - 1) // 2, 1))
        return np.squeeze(
            scipy.signal.convolve2d(
                np.concatenate(
                    [pad * data[:, None][0], data[:, None], pad * data[:, None][-1]],
                    axis=0,
                ),
                kernel,
                mode="valid",
            )
        )
    else:
        pad = np.ones(((n - 1) // 2, data.shape[1]))
        return scipy.signal.convolve2d(
            np.concatenate([pad * data[0], data, pad * data[-1]], axis=0),
            kernel,
            mode="valid",
        )


def gkern(l: int = 5, sig: float = 1.0):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / np.sum(gauss)
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def violin_scatter(
    data,
    pos=0,
    color="cornflowerblue",
    bw_method=None,
    ax=None,
    return_locs=False,
    widths=0.5,
    mark_mean=False,
    alpha=None,
):
    """plot a violin plot with scatter points jiittered around the width of the violin plot"""
    if ax is None:
        ax = plt.gca()
    vp = ax.violinplot(
        data,
        positions=[pos],
        showmedians=False,
        showextrema=False,
        points=1000,
        bw_method=bw_method,
        widths=widths,
    )
    body = vp["bodies"][0]
    body.set_facecolor(color)
    path = body.get_paths()[0].vertices
    x_data, y_data = path[:, 0], path[:, 1]
    y_data = y_data[x_data.size // 2 :]  # -pos
    x_data = x_data[x_data.size // 2 :] - pos
    width = x_data[np.digitize(data, y_data, right=False)]
    x_pos = np.random.normal(0, 0.3, len(data)) * width + pos
    alpha = 1 - (len(data)) / (len(data) + 20)
    if alpha is None:
        alpha = np.min([0.5, alpha])
        alpha = np.max([0.01, alpha])
    ax.scatter(x_pos, data, alpha=alpha, color=color)
    if mark_mean:
        ax.scatter(pos, np.mean(data), color=color)
    if return_locs:
        return x_pos, data
    return


def get_slope(data, time):
    from scipy.stats import linregress

    slope = []
    for i in range(data.shape[0]):
        slope.append(linregress(time, data[i]).slope)
    return np.array(slope)


def parse_unit_ids(unit_ids):
    """parse unit_ids from from dict to unique string for hashability"""
    if type(unit_ids[0]) is dict:
        unit_ids = [f"{x['spikesorting_merge_id']}_{x['unit_id']}" for x in unit_ids]
    return unit_ids


early_wtrack_files = [
    "Yoshi20220517_.nwb",
    "Yoshi20220518_.nwb",
    "Olive20220711_.nwb",
    "Wallie20220922_.nwb",
    "Bilbo20230802_.nwb",
    "Bilbo20230804_.nwb",
    "Totoro20220613_.nwb",
    "Totoro20220614_.nwb",
    "Winnie20220719_.nwb",
    "Banner20220224_.nwb",
    "Banner20220225_.nwb",
    "Frodo20230814_.nwb",
]

early_wtrack_keys = [dict(nwb_file_name=x) for x in early_wtrack_files]

first_wtrack_keys = [
    {"nwb_file_name": "Yoshi20220517_.nwb", "interval_list_name": "pos 1 valid times"},
    {"nwb_file_name": "Olive20220711_.nwb", "interval_list_name": "pos 1 valid times"},
    {"nwb_file_name": "Wallie20220922_.nwb", "interval_list_name": "pos 1 valid times"},
    {"nwb_file_name": "Bilbo20230802_.nwb", "interval_list_name": "pos 1 valid times"},
    {"nwb_file_name": "Totoro20220613_.nwb", "interval_list_name": "pos 1 valid times"},
    {"nwb_file_name": "Winnie20220719_.nwb", "interval_list_name": "pos 1 valid times"},
    {"nwb_file_name": "Banner20220224_.nwb", "interval_list_name": "pos 1 valid times"},
    {"nwb_file_name": "Frodo20230814_.nwb", "interval_list_name": "pos 1 valid times"},
]
