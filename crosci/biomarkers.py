# This file is part of crosci, licensed under Creative Commons Attribution-NonCommercial (CC BY-NC).
# See LICENSE.txt for more details.

import numpy as np
from numpy.matlib import repmat
from PyAstronomy.pyasl import generalizedESD
from scipy.stats import expon
from scipy.optimize import differential_evolution

def fEI(signal, sampling_frequency, window_size_sec, window_overlap, DFA_array, bad_idxes = []):
    """ Calculates fEI (on a set window size) for signal

        Steps refer to description of fEI algorithm in Figure 2D of paper:
          Measurement of excitation inhibition ratio in autism spectrum disorder using critical brain dynamics
          Scientific Reports (2020)
          Hilgo Bruining*, Richard Hardstone*, Erika L. Juarez-Martinez*, Jan Sprengers*, Arthur-Ervin Avramiea, Sonja Simpraga, Simon J. Houtman, Simon-Shlomo Poil5,
          Eva Dallares, Satu Palva, Bob Oranje, J. Matias Palva, Huibert D. Mansvelder & Klaus Linkenkaer-Hansen
          (*Joint First Author)

        Originally created by Richard Hardstone (2020), rhardstone@gmail.com
        Please note that commercial use of this algorithm is protected by Patent claim (PCT/NL2019/050167) “Method of determining brain activity”; with priority date 16 March 2018
        This code is licensed under creative commons license CC-BY-NC https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt

    Parameters
    ----------
    signal: array, shape(n_channels,n_times)
        amplitude envelope for all channels
    sampling_frequency: integer
        sampling frequency of the signal
    window_size_sec: float
        window size in seconds
    window_overlap: float
        fraction of overlap between windows (0-1)
    DFA_array: array, shape(n_channels)
        array of DFA values, with corresponding value for each channel, used for thresholding fEI
    bad_idxes: array, shape(n_channels)
        channels to ignore from computation are marked with 1, the rest with 0. can also be empty list,
        case in which all channels are computed

    Returns
    -------
    fEI_outliers_removed: array, shape(n_channels)
        fEI values, with outliers removed
    fEI_val: array, shape(n_channels)
        fEI values, with outliers included
    num_outliers: integer
        number of detected outliers
    wAmp: array, shape(n_channels, num_windows)
        windowed amplitude, computed across all channels/windows
    wDNF: array, shape(n_channels, num_windows)
        windowed detrended normalized fluctuation, computed across all channels/windows
    """

    window_size = int(window_size_sec * sampling_frequency)

    num_chans = np.shape(signal)[0]
    length_signal = np.shape(signal)[1]

    channels_to_ignore = [False] * num_chans

    for bad_idx in bad_idxes:
       channels_to_ignore[bad_idx] = True

    window_offset = int(np.floor(window_size * (1 - window_overlap)))
    all_window_index = _create_window_indices(length_signal, window_size, window_offset)
    num_windows = np.shape(all_window_index)[0]

    fEI_val = np.zeros((num_chans, 1))
    fEI_val[:] = np.NAN
    fEI_outliers_removed = np.zeros((num_chans, 1))
    fEI_outliers_removed[:] = np.NAN
    num_outliers = np.zeros((num_chans, 1))
    num_outliers[:] = np.NAN
    wAmp = np.zeros((num_chans, num_windows))
    wAmp[:] = np.NAN
    wDNF = np.zeros((num_chans, num_windows))
    wDNF[:] = np.NAN

    for ch_idx in range(num_chans):
        if channels_to_ignore[ch_idx]:
            continue

        original_amp = signal[ch_idx, :]

        if np.min(original_amp) == np.max(original_amp):
            print('Problem computing fEI for channel idx '+str(ch_idx))
            continue

        signal_profile = np.cumsum(original_amp - np.mean(original_amp))
        w_original_amp = np.mean(original_amp[all_window_index], axis=1)

        x_amp = repmat(np.transpose(w_original_amp[np.newaxis, :]), 1, window_size)
        x_signal = signal_profile[all_window_index]
        x_signal = np.divide(x_signal, x_amp)

        # Calculate local trend, as the line of best fit within the time window
        _, fluc, _, _, _ = np.polyfit(np.arange(window_size), np.transpose(x_signal), deg=1, full=True)
        # Convert to root-mean squared error, from squared error
        w_detrendedNormalizedFluctuations = np.sqrt(fluc / window_size)

        fEI_val[ch_idx] = 1 - np.corrcoef(w_original_amp, w_detrendedNormalizedFluctuations)[0, 1]

        gesd_alpha = 0.05
        max_outliers_percentage = 0.025  # this is set to 0.025 per dimension (2-dim: wAmp and wDNF), so 0.05 is max
        # smallest value for max number of outliers is 2 for generalizedESD
        max_num_outliers = max(int(np.round(max_outliers_percentage * len(w_original_amp))),2)
        outlier_indexes_wAmp = generalizedESD(w_original_amp, max_num_outliers, gesd_alpha)[1]
        outlier_indexes_wDNF = generalizedESD(w_detrendedNormalizedFluctuations, max_num_outliers, gesd_alpha)[1]
        outlier_union = outlier_indexes_wAmp + outlier_indexes_wDNF
        num_outliers[ch_idx, :] = len(outlier_union)
        not_outlier_both = np.setdiff1d(np.arange(len(w_original_amp)), np.array(outlier_union))
        fEI_outliers_removed[ch_idx] = 1 - np.corrcoef(w_original_amp[not_outlier_both], \
                                                        w_detrendedNormalizedFluctuations[not_outlier_both])[0, 1]

        wAmp[ch_idx, :] = w_original_amp
        wDNF[ch_idx, :] = w_detrendedNormalizedFluctuations

    fEI_val[DFA_array <= 0.6] = np.nan
    fEI_outliers_removed[DFA_array <= 0.6] = np.nan

    return (fEI_outliers_removed, fEI_val, num_outliers, wAmp, wDNF)


def DFA(signal, sampling_frequency, fit_interval, compute_interval, overlap=True, bad_idxes=[]):
    """ Calculates DFA of a signal

    Parameters
    ----------
    signal: array, shape(n_channels,n_times)
        amplitude envelope for all channels
    sampling_frequency: integer
        sampling frequency of the signal
    fit_interval: list, length 2
        interval (in seconds) over which the DFA exponent is fit. should be included in compute_interval
    compute_interval: list, length 2
        interval (in seconds) over which DFA is computed
    overlap: boolean
        if set to True, then windows are generated with an overlap of 50%
    bad_idxes: array, shape(n_channels)
        channels to ignore from computation are marked with 1, the rest with 0. can also be empty list,
        case in which all channels are computed

    Returns
    -------
    dfa_array, window_sizes, fluctuations, dfa_intercept
    dfa_array: array, shape(n_channels)
        DFA value for each channel
    window_sizes: array, shape(num_windows)
        window sizes over which the fluctuation function is computed
    fluctuations: array, shape(num_windows)
        fluctuation function value at each computed window size
    dfa_intercept: array, shape(n_channels)
        DFA intercept for each channel
    """

    num_chans, num_timepoints = np.shape(signal)

    channels_to_ignore = [False] * num_chans
    for bad_idx in bad_idxes:
        channels_to_ignore[bad_idx] = True

    length_signal = np.shape(signal)[1]

    assert fit_interval[0] >= compute_interval[0] and fit_interval[1] <= compute_interval[
        1], 'CalcInterval should be included in ComputeInterval'
    assert compute_interval[0] >= 0.1 and compute_interval[
        1] <= 1000, 'ComputeInterval should be between 0.1 and 1000 seconds'
    assert compute_interval[1]/sampling_frequency <= num_timepoints, \
        'ComputeInterval should not extend beyond the length of the signal'

    # compute DFA window sizes for the given CalcInterval
    window_sizes = np.floor(np.logspace(-1, 3, 81) * sampling_frequency).astype(
        int)  # %logspace from 0.1 seccond (10^-1) to 1000 (10^3) seconds

    # make sure there are no duplicates after rounding
    window_sizes = np.sort(np.unique(window_sizes))

    window_sizes = window_sizes[(window_sizes >= compute_interval[0] * sampling_frequency) & \
                                (window_sizes <= compute_interval[1] * sampling_frequency)]

    dfa_array = np.zeros(num_chans)
    dfa_array[:] = np.NAN
    dfa_intercept = np.zeros(num_chans)
    dfa_intercept[:] = np.NAN
    fluctuations = np.zeros((num_chans, len(window_sizes)))
    fluctuations[:] = np.NAN

    if max(window_sizes) <= num_timepoints:
        for ch_idx in range(num_chans):
            if channels_to_ignore[ch_idx]:
                continue

            signal_for_channel = signal[ch_idx, :]

            for i_window_size in range(len(window_sizes)):
                if overlap == True:
                    window_overlap = 0.5
                else:
                    window_overlap = 0

                window_size = window_sizes[i_window_size]
                window_offset = np.floor(window_size * (1 - window_overlap))
                all_window_index = _create_window_indices(length_signal, window_sizes[i_window_size], window_offset)
                # First we convert the time series into a series of fluctuations y(i) around the mean.
                demeaned_signal = signal_for_channel - np.mean(signal_for_channel)
                # Then we integrate the above fluctuation time series ('y').
                signal_profile = np.cumsum(demeaned_signal)

                x_signal = signal_profile[all_window_index]

                # Calculate local trend, as the line of best fit within the time window -> fluc is the sum of squared residuals
                _, fluc, _, _, _ = np.polyfit(np.arange(window_size), np.transpose(x_signal), deg=1, full=True)

                # Peng's formula - Convert to root-mean squared error, from squared error
                # det_fluc = np.sqrt(np.mean(fluc / window_size))
                # Richard's formula
                det_fluc = np.mean(np.sqrt(fluc / window_size))
                fluctuations[ch_idx, i_window_size] = det_fluc

            # get the positions of the first and last window sizes used for fitting
            fit_interval_first_window = np.argwhere(window_sizes >= fit_interval[0] * sampling_frequency)[0][0]
            fit_interval_last_window = np.argwhere(window_sizes <= fit_interval[1] * sampling_frequency)[-1][0]

            # take the previous to the first window size if the difference between the lower end of fitting and
            # the previous window is no more than 1% of the lower end of fitting and if the difference between the lower
            # end of fitting and the previous window is less than the difference between the lower end of fitting and the current first window
            if np.abs(window_sizes[fit_interval_first_window-1] / sampling_frequency - fit_interval[0]) <= fit_interval[0] / 100:
                if np.abs(window_sizes[fit_interval_first_window-1] / sampling_frequency - fit_interval[0]) < \
                    np.abs(window_sizes[fit_interval_first_window] / sampling_frequency - fit_interval[0]):
                    fit_interval_first_window = fit_interval_first_window - 1

            x = np.log10(window_sizes[fit_interval_first_window:fit_interval_last_window+1])
            y = np.log10(fluctuations[ch_idx, fit_interval_first_window:fit_interval_last_window+1])
            model = np.polyfit(x, y, 1)
            dfa_intercept[ch_idx] = model[1]
            dfa_array[ch_idx] = model[0]

    return (dfa_array, window_sizes, fluctuations, dfa_intercept)


def get_frequency_bins(frequency_range):
    """ Get frequency bins for the frequency range of interest.

    Parameters
    ----------
    frequency_range : array, shape (1,2)
        The frequency range over which to create frequency bins.
        The lower edge should be equal or more than 1 Hz, and the upper edge should be equal or less than 150 Hz.

    Returns
    -------
    frequency_bins : list, shape (n_bins,2)
        The lower and upper range in Hz per frequency bin.
    """

    assert frequency_range[0] >= 1.0 and frequency_range[1] <= 150.0, \
        'The frequency range should cannot be less than 1 Hz or more than 150 Hz'

    frequency_bin_delta = [1.0, 4.0]
    frequency_range_full = [frequency_bin_delta[1], 150]
    n_bins_full = 16

    # Create logarithmically-spaced bins over the full frequency range
    frequencies_full = np.logspace(np.log10(frequency_range_full[0]), np.log10(frequency_range_full[-1]), n_bins_full)
    frequencies = np.append(frequency_bin_delta[0],frequencies_full)
    # Get frequencies that fall within the frequency range of interest
    myfrequencies = frequencies[np.where((np.round(frequencies, 4) >= frequency_range[0]) & (
                np.round(frequencies, 4) <= frequency_range[1]))[0]]

    # Get all frequency bin ranges
    frequency_bins = [[myfrequencies[i], myfrequencies[i + 1]] for i in range(len(myfrequencies) - 1)]

    n_bins = len(frequency_bins)

    return frequency_bins


def get_DFA_fitting_interval(frequency_interval):
    """ Get a fitting interval for DFA computation.

    Parameters
    ----------
    frequency_interval : array, shape (1,2)
        The lower and upper bound of the frequency bin in Hz for which the fitting interval will be inferred.
        The fitting interval is where the regression line is fit for log-log coordinates of the fluctuation function vs. time windows.

    Returns
    -------
    fit_interval : array, shape (1,2)
        The lower and upper bound of the fitting range in seconds for a frequency bin.
    """

    # Upper fitting margin in seconds
    upper_fit = 30
    # Default lower fitting margins in seconds per frequency bin
    default_lower_fits = [5., 5., 5., 3.981, 3.162, 2.238, 1.412, 1.122, 0.794, 0.562, 0.398, 0.281, 0.141, 0.1, 0.1, 0.1]

    frequency_bins = get_frequency_bins([1, 150])
    # Find the fitting interval. In case when frequency range is not exactly one from the defined frequency bins,
    # it finds the fitting interval of the bin for which the lowest of the provided frequencies falls into.
    idx_freq = np.where((np.array(frequency_bins)[:, 0] <= frequency_interval[0]))[0][-1]

    fit_interval = [default_lower_fits[idx_freq],upper_fit]

    return fit_interval


def _create_window_indices(length_signal, length_window, window_offset):

    window_starts = np.arange(0,length_signal-length_window,window_offset)
    num_windows = len(window_starts)

    one_window_index = np.arange(0,length_window)
    all_window_index = repmat(one_window_index,num_windows,1).astype(int)

    all_window_index = all_window_index + repmat(np.transpose(window_starts[np.newaxis,:]),1,length_window).astype(int)

    return all_window_index
