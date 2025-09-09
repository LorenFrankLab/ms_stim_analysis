import numpy as np


def check_confidence_interval(confidence_interval, allow_small_values=False):
    if len(confidence_interval) != 2:
        raise Exception(f"Confidence interval must have two elements")
    if np.diff(confidence_interval) < 0:
        raise Exception(
            f"Second element of confidence interval must be greater than first"
        )
    if np.logical_or(confidence_interval[0] < 0, confidence_interval[1] > 100):
        raise Exception(f"Confidence interval must be on [0 100]")
    if (
        np.max(confidence_interval) < 1 and not allow_small_values
    ):  # check that confidence interval reasonable
        raise Exception(
            "Upper bound in confidence interval is less than one; this may be an error."
            "Percentiles go from 0 to 100, not 0 to 1."
        )


def return_confidence_interval(x, alpha=0.05, allow_small_values=False):
    confidence_interval = alpha_to_percentage_confidence_interval(alpha)
    check_confidence_interval(
        confidence_interval, allow_small_values
    )  # ensure valid confidence interval
    return np.percentile(x, confidence_interval, axis=0)


def alpha_to_percentage_confidence_interval(alpha):
    return np.asarray([alpha, 1 - alpha]) * 100


def average_confidence_interval(
    x, num_bootstrap_samples=1000, alpha=0.05, average_function=None, exclude_nan=False
):
    if average_function is None:
        average_function = np.mean
    if exclude_nan:
        x = np.asarray(x)
        x = x[np.invert(np.isnan(x))]
    rng = np.random.default_rng()
    x_boot = rng.choice(x, replace=True, size=(num_bootstrap_samples, len(x)))
    x_ave_boot = average_function(x_boot, axis=1)
    return return_confidence_interval(x_ave_boot, alpha)
