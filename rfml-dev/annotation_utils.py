import cupy
from cupyx.scipy.signal import spectrogram as cupyx_spectrogram
from cupyx.scipy.ndimage import gaussian_filter as cupyx_gaussian_filter

from spectrogram import *

import data as data_class
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from tqdm import tqdm

def moving_average(complex_iq, window_len):
    return (
        np.convolve(np.abs(complex_iq) ** 2, np.ones(window_len), "valid") / window_len
    )

def power_squelch(iq_samples, threshold, window):
    avg_pwr = moving_average(iq_samples, window)
    avg_pwr_db = 10 * np.log10(avg_pwr)

    good_samples = np.zeros(len(iq_samples))
    good_samples[np.where(avg_pwr_db > threshold)] = 1

    idx = (
        np.ediff1d(np.r_[0, good_samples == 1, 0]).nonzero()[0].reshape(-1, 2)
    )  # gets indices where signal power above threshold

    return idx


def annotate_power_squelch(data_obj, threshold, avg_window_len, label=None, skip_validate=False, estimate_frequency=False, dry_run=False, min_annotation_length=400):
    iq_samples = data_obj.get_samples()
    idx = power_squelch(iq_samples, threshold=threshold, window=avg_window_len)

    data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
    for start, stop in tqdm(idx):
        start, stop = int(start), int(stop)
        if stop-start < min_annotation_length:
            continue
            
        if estimate_frequency:
            freq_lower_edge, freq_upper_edge = get_occupied_bandwidth(iq_samples[start:stop], data_obj.metadata["global"]["core:sample_rate"], data_obj.metadata["captures"][0]["core:frequency"] )
        else: 
            freq_lower_edge = data_obj.metadata["captures"][0]["core:frequency"] - data_obj.metadata["global"]["core:sample_rate"] / 2
            freq_upper_edge = data_obj.metadata["captures"][0]["core:frequency"] + data_obj.metadata["global"]["core:sample_rate"] / 2
        metadata = {
            "core:freq_lower_edge": freq_lower_edge,
            "core:freq_upper_edge": freq_upper_edge,
        }
        if label:
            metadata["core:label"] = label
        data_obj.sigmf_obj.add_annotation(start, length=stop - start, metadata=metadata)

    # print(f"{data_obj.sigmf_obj=}")

    if not dry_run: 
        data_obj.sigmf_obj.tofile(data_obj.sigmf_meta_filename, skip_validate=skip_validate)
        print(f"Writing {len(data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY])} annotations to {data_obj.sigmf_meta_filename}")


def annotate(filename, label, avg_window_len, avg_duration=-1, debug=False, dry_run=False):
    
    data_obj = data_class.Data(filename)

    # use a seconds worth of data to calculate threshold
    if avg_duration > -1:
        iq_samples = data_obj.get_samples(n_samples=int(data_obj.metadata["global"]["core:sample_rate"]*avg_duration))
    else:
        iq_samples = data_obj.get_samples()
        
    avg_pwr = moving_average(iq_samples, avg_window_len)
    avg_pwr_db = 10*np.log10(avg_pwr)


    # current threshold in custom_handler 
    guess_threshold_old = (np.max(avg_pwr_db) + np.mean(avg_pwr_db))/2

    # MAD estimator
    def median_absolute_deviation(series):
        mad = 1.4826 * np.median(np.abs(series - np.median(series)))
        # sci_mad = scipy.stats.median_abs_deviation(series, scale="normal")
        return np.median(series) + 6*mad

    mad = median_absolute_deviation(avg_pwr_db)
    guess_threshold = mad
    
    if debug:
        print(f"{np.max(avg_pwr_db)=}")
        print(f"{np.mean(avg_pwr_db)=}")
        print(f"median absolute deviation threshold = {mad}")
        print(f"using threshold = {guess_threshold}")
        # print(f"{len(avg_pwr_db)=}")
        
        plt.figure()
        db_plot = avg_pwr_db[int(0*20.48e6):int(avg_duration*20.48e6)]
        plt.plot(np.arange(len(db_plot))/data_obj.metadata["global"]["core:sample_rate"], db_plot)
        plt.axhline(y = guess_threshold_old, color = 'g', linestyle = '-', label="old threshold") 
        plt.axhline(y = np.mean(avg_pwr_db), color = 'r', linestyle = '-', label="average") 
        plt.axhline(y = mad, color = 'b', linestyle = '-', label="median absolute deviation threshold") 
        plt.legend(loc="upper left")
        plt.ylabel("dB")
        plt.xlabel("time (seconds)")
        plt.title("Signal Power")
        plt.show()

    annotate_power_squelch(data_obj, guess_threshold, avg_window_len, label=label, skip_validate=True, estimate_frequency=True, dry_run=dry_run)
def get_occupied_bandwidth(samples, sample_rate, center_frequency):

    # spectrogram_data, spectrogram_raw = spectrogram(
    #     samples,
    #     sample_rate,
    #     256,
    #     0,
    # )
    # spectrogram_color = spectrogram_cmap(spectrogram_data, plt.get_cmap("viridis"))

    # plt.figure()
    # plt.imshow(spectrogram_color)
    # plt.show()

    # print(f"{samples.shape=}")
    # print(f"{samples=}")
    
    f, t, Sxx = cupyx_spectrogram(samples, fs=sample_rate, return_onesided=False, scaling="spectrum")

    freq_power = cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0))
    # print(f"{freq_power.shape=}")

    # print(f"{freq_power.argmax(axis=0).shape=}")
    # print(f"{freq_power.argmax(axis=0)=}")
    
    # freq_power = cupy.asnumpy(cupyx_gaussian_filter(cupy.fft.fftshift(Sxx, axes=0), sigma=1))
    # freq_power = cupyx_gaussian_filter(cupy.fft.fftshift(Sxx, axes=0), sigma=1)
    freq_power = cupy.median(cupy.fft.fftshift(Sxx, axes=0), axis=1)
    # plt.figure()
    # plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # plt.figure()
    # # plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(freq_power))
    # plt.imshow(np.tile(np.expand_dims(cupy.asnumpy(freq_power), axis=1), (1,Sxx.shape[1])))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # print(f"{freq_power.shape=}")

    # print(f"{freq_power.argmax(axis=0).shape=}")
    # print(f"{freq_power.argmax(axis=0)=}")
    freq_power_normalized = freq_power / freq_power.sum(axis=0)

    # print(f"{freq_power_normalized.shape=}")
    # print(f"{freq_power_normalized.argmax(axis=0).shape=}")
    # print(f"{freq_power_normalized.argmax(axis=0)=}")
    # freq_power_max_idxs = freq_power_normalized.argmax(axis=0)
    # if freq_power_max_idxs.size == 1:
    #     freq_power_max_idxs = np.expand_dims(cupy.asnumpy(freq_power_max_idxs), axis=0)
    # print(f"{freq_power_max_idxs=}")
    # print(f"{freq_power_max_idxs.shape=}")
    # for i, max_power_idx in enumerate(freq_power_max_idxs):

        
        # max_power_idx = int(cupy.asnumpy(max_power_idx))
        # print(f"{i=}, {max_power_idx=}")
    max_power_idx = int(cupy.asnumpy(freq_power_normalized.argmax(axis=0)))
    lower_idx = max_power_idx
    upper_idx = max_power_idx
    while True:

        if upper_idx == freq_power_normalized.shape[0]-1:
            lower_idx -= 1
        elif lower_idx == 0: 
            upper_idx += 1
        elif freq_power_normalized[lower_idx] > freq_power_normalized[upper_idx]: 
            lower_idx -= 1
        else: 
            upper_idx += 1
        
        # print(f"{lower_idx=}, {upper_idx=}")
        # print(f"{freq_power_normalized[lower_idx:upper_idx, i].sum()=}")
        if freq_power_normalized[lower_idx:upper_idx].sum() >= 0.94:
            break
                
    bounds = np.array([lower_idx,upper_idx])
        
        
    # plt.figure()
    # plt.imshow(cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))

    # plt.axhline(y = bounds[0], color = 'r', linestyle = '-') 
    # plt.axhline(y = bounds[1], color = 'b', linestyle = '-') 
    # plt.show()
    
                              
    freq_lower_edge = center_frequency + (freq_power.shape[0]/2 - bounds[1])/freq_power.shape[0]*sample_rate
    freq_upper_edge = center_frequency + (freq_power.shape[0]/2 - bounds[0])/freq_power.shape[0]*sample_rate

    # print(f"{freq_lower_edge=}")
    # print(f"{freq_upper_edge=}")
    # print(f"estimated bandwidth = {freq_upper_edge-freq_lower_edge}")
    # raise ValueError
    return freq_lower_edge, freq_upper_edge
    
def get_occupied_bandwidth_backup(samples, sample_rate, center_frequency):

    # spectrogram_data, spectrogram_raw = spectrogram(
    #     samples,
    #     sample_rate,
    #     256,
    #     0,
    # )
    # spectrogram_color = spectrogram_cmap(spectrogram_data, plt.get_cmap("viridis"))

    # plt.figure()
    # plt.imshow(spectrogram_color)
    # plt.show()

    # print(f"{samples.shape=}")
    # print(f"{samples=}")
    
    f, t, Sxx = cupyx_spectrogram(samples, fs=sample_rate, return_onesided=False, scaling="spectrum")

    freq_power = cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0))
    # print(f"{freq_power.shape=}")

    # print(f"{freq_power.argmax(axis=0).shape=}")
    # print(f"{freq_power.argmax(axis=0)=}")
    
    # freq_power = cupy.asnumpy(cupyx_gaussian_filter(cupy.fft.fftshift(Sxx, axes=0), sigma=1))
    freq_power = cupyx_gaussian_filter(cupy.fft.fftshift(Sxx, axes=0), sigma=1)

    # plt.figure()
    # plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # plt.figure()
    # plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(cupy.fft.fftshift(f)), cupy.asnumpy(freq_power))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    
    freq_power_normalized = freq_power / freq_power.sum(axis=0)

    # print(f"{freq_power_normalized.shape=}")
    # print(f"{freq_power_normalized.argmax(axis=0).shape=}")
    # print(f"{freq_power_normalized.argmax(axis=0)=}")
    bounds = []
    for i, max_power_idx in enumerate(freq_power_normalized.argmax(axis=0)):
        max_power_idx = int(cupy.asnumpy(max_power_idx))
        # print(f"{i=}, {max_power_idx=}")
        lower_idx = max_power_idx
        upper_idx = max_power_idx
        while True:

            if upper_idx == freq_power_normalized.shape[0]-1:
                lower_idx -= 1
            elif lower_idx == 0: 
                upper_idx += 1
            elif freq_power_normalized[lower_idx,i] > freq_power_normalized[upper_idx,i]: 
                lower_idx -= 1
            else: 
                upper_idx += 1
            
            # print(f"{lower_idx=}, {upper_idx=}")
            # print(f"{freq_power_normalized[lower_idx:upper_idx, i].sum()=}")
            if freq_power_normalized[lower_idx:upper_idx, i].sum() >= 0.94:
                break
                
        bounds.append([lower_idx,upper_idx])
    bounds = np.array(bounds)
        
        
    plt.figure()
    plt.imshow(cupy.asnumpy(cupy.fft.fftshift(Sxx, axes=0)))
    plt.plot(cupy.asnumpy(freq_power.argmax(axis=0)))
    plt.plot(bounds[:,0])
    plt.plot(bounds[:,1])
    plt.axhline(y = np.median(bounds[:,0]), color = 'r', linestyle = '-') 
    plt.axhline(y = np.median(bounds[:,1]), color = 'b', linestyle = '-') 
    plt.show()
    
                              
    freq_lower_edge = center_frequency + (freq_power.shape[0]/2 - np.median(bounds[:,1]))/freq_power.shape[0]*sample_rate
    freq_upper_edge = center_frequency + (freq_power.shape[0]/2 - np.median(bounds[:,0]))/freq_power.shape[0]*sample_rate

    # print(f"{freq_lower_edge=}")
    # print(f"{freq_upper_edge=}")
    print(f"estimated bandwidth = {freq_upper_edge-freq_lower_edge}")
    return freq_lower_edge, freq_upper_edge


def reset_predictions_sigmf(dataset):
    data_files = set([dataset.index[i][1].absolute_path for i in range(len(dataset))])
    for f in data_files: 
        data_obj = data_class.Data(f)
        prediction_meta_path = Path(Path(data_obj.sigmf_meta_filename).parent, f"prediction_{Path(data_obj.sigmf_meta_filename).name}")
        data_obj.sigmf_obj._metadata[data_obj.sigmf_obj.ANNOTATION_KEY] = []
        data_obj.sigmf_obj.tofile(prediction_meta_path, skip_validate=True)
        print(f"Reset annotations in {prediction_meta_path}")
