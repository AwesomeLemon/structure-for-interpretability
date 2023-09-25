import scipy
import scipy.stats
import numpy as np


def wasserstein_divergence(u_values, v_values, u_weights=None, v_weights=None):
        # Adapted from scipy.stats.wasserstein_distance and _cdf_distance
        u_values, u_weights = scipy.stats.stats._validate_distribution(u_values, u_weights)
        v_values, v_weights = scipy.stats.stats._validate_distribution(v_values, v_weights)

        u_sorter = np.argsort(u_values)
        v_sorter = np.argsort(v_values)

        all_values = np.concatenate((u_values, v_values))
        all_values.sort(kind='mergesort')

        # Compute the differences between pairs of successive values of u and v.
        deltas = np.diff(all_values)

        # Get the respective positions of the values of u and v among the values of
        # both distributions.
        u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
        v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

        # Calculate the CDFs of u and v using their weights, if specified.
        if u_weights is None:
            u_cdf = u_cdf_indices / u_values.size
        else:
            u_sorted_cumweights = np.concatenate(([0], np.cumsum(u_weights[u_sorter])))
            u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

        if v_weights is None:
            v_cdf = v_cdf_indices / v_values.size
        else:
            v_sorted_cumweights = np.concatenate(([0],
                                                np.cumsum(v_weights[v_sorter])))
            v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

        # Compute the value of the integral based on the CDFs.
        # Removes the np.abs to compute a divergence.
        return np.sum(np.multiply(u_cdf - v_cdf, deltas))


def kolmogorov_smirnov_difference(u_values, v_values, u_weights=None, v_weights=None, return_position=False):
        # Adapted from scipy.stats.wasserstein_distance and _cdf_distance
        u_values, u_weights = scipy.stats._stats_py._validate_distribution(u_values, u_weights)
        v_values, v_weights = scipy.stats._stats_py._validate_distribution(v_values, v_weights)

        u_sorter = np.argsort(u_values)
        v_sorter = np.argsort(v_values)

        all_values = np.concatenate((u_values, v_values))
        all_values.sort(kind='mergesort')

        # Get the respective positions of the values of u and v among the values of
        # both distributions.
        u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
        v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

        # Calculate the CDFs of u and v using their weights, if specified.
        if u_weights is None:
            u_cdf = u_cdf_indices / u_values.size
        else:
            u_sorted_cumweights = np.concatenate(([0], np.cumsum(u_weights[u_sorter])))
            u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

        if v_weights is None:
            v_cdf = v_cdf_indices / v_values.size
        else:
            v_sorted_cumweights = np.concatenate(([0],
                                                np.cumsum(v_weights[v_sorter])))
            v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

        # Compute the value of the integral based on the CDFs.
        # Removes the np.abs to compute a divergence.

        diffs = u_cdf - v_cdf
        d_arr = np.array([diffs.min(), diffs.max()])
        ksd = d_arr[np.argmax(np.abs(d_arr))]

        if return_position:
            if np.argmax(np.abs(d_arr)) == 0:
                x_ind = diffs.argmin()
            else:
                x_ind = diffs.argmax()
            x_val = all_values[x_ind]
            return ksd, x_val

        return ksd


class DistributionDifference:
    def __init__(self):
        pass

    def compute(self, d_out_class, d_in_class):
        pass

    def save_name(self):
        pass

    def pretty_name(self):
        pass


class Shift(DistributionDifference):
    "This the mean shift and the same as the Wasserstein-1 Distance without the absolute value."
    def __init__(self):
        super().__init__()

    def compute(self, d_out_class, d_in_class):
        return wasserstein_divergence(d_out_class, d_in_class)
    
    def save_name(self):
        return "shift"
    
    def pretty_name(self):
        return "Shift"
    

class KolmogorovSmirnovDifference(DistributionDifference):
    def __init__(self):
        super().__init__()

    def compute(self, d_out_class, d_in_class):
        return kolmogorov_smirnov_difference(d_out_class, d_in_class)

    def save_name(self):
        return "ks_diff"
    
    def pretty_name(self):
        return "Kolmogorov-Smirnov Diff."
    

class ModifiedWassersteinDistance(DistributionDifference):
    def __init__(self):
        super().__init__()

    def compute(self, d_out_class, d_in_class):
        min_out, max_out = d_out_class.min(), d_out_class.max()
        min_in, max_in = d_in_class.min(), d_in_class.max()
        values_min = min(min_out, min_in)
        values_max = max(max_out, max_in)

        wd_normed = scipy.stats.wasserstein_distance(
            (d_out_class - values_min) / (values_max - values_min),
            (d_in_class - values_min) / (values_max - values_min))

        # this means: 2 to the right of 1 -> positive, 2 to the left of 1 -> negative
        wd_normed *= np.sign(d_in_class.mean() - d_out_class.mean())
        return wd_normed

    def save_name(self):
        return "dist"

    def pretty_name(self):
        return "MWD"