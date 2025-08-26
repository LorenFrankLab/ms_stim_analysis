import numpy as np


def normalize_by_index_wrapper(index_norms):
    def normalize_by_index(arr, index, **kwargs):
        return arr / index_norms[index]

    return normalize_by_index


def normalize_by_peak(arr, **kwargs):
    return arr / np.max(np.abs(arr))


def shuffled_trace_distribution(
    marks,
    signal,
    time,
    marks_id=None,  # can be used to track cycle#, pulse#, etc.
    n_shuffles=500,
    shuffle_window=0.125,
    sample_window=(-125, 125),
    n_samples=1000,
    normalize_func=normalize_by_peak,
    measurements=None,
):
    dt = np.mean(np.diff(time))
    if marks_id is None:
        marks_id = np.zeros_like(marks)

    shuffled_marks = []
    shuffled_marks_id = []
    orig_ind = np.digitize(marks, time)
    valid_marks = np.where(
        (orig_ind - shuffle_window / dt / 2 + sample_window[0] > 0)
        & (orig_ind + shuffle_window / dt / 2 + sample_window[1] < len(time))
    )[0]
    orig_ind = orig_ind[valid_marks]
    marks_id = marks_id[valid_marks]
    for _ in range(n_shuffles):
        shuffled_marks.extend(
            orig_ind
            + np.random.randint(-shuffle_window / dt / 2, shuffle_window / dt / 2)
        )
        shuffled_marks_id.extend(marks_id)

    if measurements is None:
        # just return the full set of normalized traces
        return [
            normalize_func(
                signal[m + sample_window[0] : m + sample_window[1]], index=id
            )
            for m, id in zip(shuffled_marks, shuffled_marks_id)
        ]

    shuffled_marks = np.array(shuffled_marks)
    bootstrap_dist = [[] for _ in measurements]
    for _ in range(n_samples):
        ind = np.random.choice(len(shuffled_marks), len(marks))
        sample_set = [
            normalize_func(
                signal[
                    shuffled_marks[i]
                    + sample_window[0] : shuffled_marks[i]
                    + sample_window[1]
                ],
                index=shuffled_marks_id[i],
            )
            for i in ind
        ]
        for i, M in enumerate(measurements):
            bootstrap_dist[i].append(M(sample_set))
    return bootstrap_dist


def shuffled_spiking_distribution(
    marks, alligned_binned_spike_func, n_shuffles=500, shuffle_window=0.125
):
    shuffled_counts = []
    for i in range(n_shuffles):
        shuffled_counts.append(
            alligned_binned_spike_func(
                marks + np.random.uniform(-shuffle_window / 2, shuffle_window / 2)
            )
        )
    shuffled_counts = np.concatenate(
        shuffled_counts, axis=1
    )  # concatenate along the marks axis
    return shuffled_counts  # shape = (units, marks, bins)


def bootstrap(
    samples,
    measurement,
    n_samples,  # number of samples to draw from the original dataset
    n_boot,  # number of bootstrap iterations
):
    bootstrap_dist = []
    for _ in range(n_boot):
        ind = np.random.choice(len(samples), n_samples)
        sample_set = samples[ind]  # [samples[i] for i in ind]
        bootstrap_dist.append(measurement(sample_set))
    return np.nanmean(bootstrap_dist, axis=0), (
        np.nanpercentile(bootstrap_dist, 0.5, axis=0),
        np.nanpercentile(bootstrap_dist, 99.5, axis=0),
    )


"""
Measurements
"""


def trace_median(arr):
    return np.median(arr, axis=0)


def count_distribution(arr):
    return np.sum(arr, axis=0) / np.sum(arr)


def discrete_KL_divergence(p, q="uniform", laplace_smooth=True, pool_bins=5):
    """Calculates the discrete KL divergence between two distributions
    Options: q = 'uniform' or a numpy array of the same shape as p
    laplace_smooth: bool, whether to add 1 to all counts to avoid log(0)
    """
    p = p / p.sum()
    if q == "uniform":
        q = np.ones(p.shape) / p.shape[0]
    if pool_bins > 1:
        p = np.sum(p[: p.size - p.size % pool_bins].reshape(-1, pool_bins), axis=1)
        p = p / p.sum()
        q = np.sum(q[: q.size - q.size % pool_bins].reshape(-1, pool_bins), axis=1)
        q = q / q.sum()
    if laplace_smooth:
        p = p + 1
        p = p / p.sum()  # / (p.sum() + p.shape[0])
        q = q + 1
        q = q / q.sum()  # / (q.sum() + q.shape[0])
    return np.nansum(p * np.log2(p.astype(float) / q.astype(float)))


def stacked_marks_to_kl(arr):
    arr = count_distribution(arr)
    return discrete_KL_divergence(arr)
