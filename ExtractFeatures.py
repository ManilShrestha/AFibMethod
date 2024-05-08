import numpy as np
import numpy as np
from scipy import stats
from scipy.fft import fft
from scipy.stats import entropy
import scipy.signal as signal
import nolds
from tqdm import tqdm
from Utilities import * 
from concurrent.futures import ThreadPoolExecutor


class ExtractFeatures:
    def __init__(self, X) -> None:
        self.X = X


    def get_features(self):
        time_domain_stats = self.compute_time_domain_statistics(self.X)
        time_domain_stats_arr = np.column_stack(list(time_domain_stats.values()))

        freq_domain_stats = self.compute_freq_domain_statistics(self.X)
        freq_domain_stats_arr = np.column_stack(list(freq_domain_stats.values()))

        B2Bintervals = self.compute_B2BInterval(self.X)
        poincare_features = self.compute_poincare_features(B2Bintervals)
        poincare_features_arr = np.column_stack(list(poincare_features.values()))

        B2B_diff_features = self.compute_b2b_diff_features(self.X)
        B2B_diff_features_arr = np.column_stack(list(B2B_diff_features.values()))

        features = np.hstack((time_domain_stats_arr, freq_domain_stats_arr, poincare_features_arr, B2B_diff_features_arr))
        
        # Zero out the nans, infs to ensure numerical stability
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


    
    def compute_time_domain_statistics(self, data):
        mean = np.mean(data, axis=1)
        median = np.median(data, axis=1)
        std_dev = np.std(data, axis=1)
        variance = np.var(data, axis=1)
        iqr = stats.iqr(data, axis=1)
        skewness = stats.skew(data, axis=1)
        scaled_data = data * 1e6  # Scale up the data to avoid precision issues
        kurtosis = stats.kurtosis(scaled_data, axis=1)
        rms = np.sqrt(np.mean(np.square(data), axis=1))

        def shannon_entropy(signaldata):
            # Normalize the signal
            s = np.sum(signaldata)
            prob_density = signaldata / (1 if s is None else s)
            # Use scipy's entropy function to calculate Shannon entropy
            return entropy(prob_density, base=2)
        
        shannon_entropy_values = np.apply_along_axis(shannon_entropy, 1, data)
        first_derivative = np.diff(data, axis=1)
        mean_first_derivative = np.mean(first_derivative, axis=1)
        std_dev_first_derivative = np.std(first_derivative, axis=1)
        
        # Return a dictionary with all results
        return {
            'mean': mean,
            'median': median,
            'standard_deviation': std_dev,
            'variance': variance,
            'interquartile_range': iqr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'root_mean_square': rms,
            'shannon_entropy': shannon_entropy_values,
            'mean_first_derivative': mean_first_derivative,
            'std_dev_first_derivative': std_dev_first_derivative
        }


    def compute_freq_domain_statistics(self, data, sample_rate = 125):
        features = {
            'first_moment': [],
            'second_moment': [],
            'third_moment': [],
            'fourth_moment': [],
            'median_frequency': [],
            'spectral_entropy': [],
            'total_spectral_power': [],
            'peak_amplitude': []
        }
        
        for signaldata in data:
            # Compute the FFT
            freq_data = fft(signaldata)
            # Get the power spectrum
            power_spectrum = np.abs(freq_data)**2
            # Get the frequencies for bins
            freqs = np.fft.fftfreq(len(signaldata), 1/sample_rate)
            
            # Consider only positive frequencies
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            power_spectrum = power_spectrum[pos_mask]
            
            # Moments of the power spectrum
            total_power = np.sum(power_spectrum)
            average_power = power_spectrum / total_power
            features['first_moment'].append(np.sum(freqs * average_power))
            features['second_moment'].append(np.sum((freqs**2) * average_power))
            features['third_moment'].append(np.sum((freqs**3) * average_power))
            features['fourth_moment'].append(np.sum((freqs**4) * average_power))
            
            # Median frequency
            cumulative_power = np.cumsum(average_power)
            median_freq_index = np.where(cumulative_power >= 0.5)[0][0]
            features['median_frequency'].append(freqs[median_freq_index])
            
            # Spectral entropy
            features['spectral_entropy'].append(entropy(average_power))
            
            # Total spectral power
            features['total_spectral_power'].append(total_power)
            
            # Peak amplitude in 0 to 10 Hz
            relevant_mask = (freqs >= 0) & (freqs <= 10)
            peak_amplitude = np.max(power_spectrum[relevant_mask]) if np.any(relevant_mask) else 0
            features['peak_amplitude'].append(peak_amplitude)
        
        # Convert lists to numpy arrays for easier handling later
        for key in features:
            features[key] = np.array(features[key])

        return features
    
    # Beat to Beat Analysis
    def detect_peaks(self, signaldata, sampling_rate=125):
        # Use a peak detection algorithm, scipy's find_peaks could be suitable
        min_distance = int(sampling_rate * 0.3)  
        peaks, _ = signal.find_peaks(signaldata, distance=min_distance)  # distance at least around 3 seconds

        return peaks


    def compute_intervals(self, peaks, sampling_rate):
        # Convert peak indices to time intervals in seconds
        intervals = np.diff(peaks) / sampling_rate
        return intervals


    def compute_B2BInterval(self, data, sampling_rate=125):
        results = []
        for signaldata in data:
            peaks = self.detect_peaks(signaldata, sampling_rate)
            intervals = self.compute_intervals(peaks, sampling_rate)
            results.append(intervals)
        return results


    def calculate_poincare_features(self, intervals):
        """
        Calculate the Poincaré plot features SD1 and SD2 for a given set of intervals.
        
        Args:
        intervals (np.array): numpy array of beat-to-beat intervals
        
        Returns:
        dict: dictionary with 'SD1', 'SD2', and 'SD1_SD2_ratio'
        """
        
        if len(intervals) < 2:
            return {'SD1': np.nan, 'SD2': np.nan, 'SD1_SD2_ratio': np.nan}

        # Calculate differences between consecutive intervals
        diff_intervals = np.diff(intervals)
        
        # Compute SD1 and SD2
        SD1 = np.sqrt(np.var(diff_intervals, ddof=1) / 2)
        SD2 = np.sqrt(2 * np.var(intervals, ddof=1) - (np.var(diff_intervals, ddof=1) / 2))
        
        # Compute the SD1/SD2 ratio
        SD1_SD2_ratio = SD1 / SD2
        
        return {'SD1': SD1, 'SD2': SD2, 'SD1_SD2_ratio': SD1_SD2_ratio}


    def compute_poincare_features(self, interval_data):
        SD1 = []
        SD2 = []
        SD1_SD2_ratio = []

        for intervals in interval_data:
            result = self.calculate_poincare_features(intervals)
            SD1.append(result['SD1'])
            SD2.append(result['SD2'])
            SD1_SD2_ratio.append(result['SD1_SD2_ratio'])
        
        return {
            'SD1': np.array(SD1),
            'SD2': np.array(SD2),
            'SD1_SD2_ratio': np.array(SD1_SD2_ratio)
        }

    def extract_b2b_segments(self, data_signal, sampling_rate=125):
        # Detect peaks and extract segments between them
        peaks = self.detect_peaks(data_signal, sampling_rate)
        segments = [data_signal[peaks[i-1]:peaks[i]] for i in range(1, len(peaks))]
        return segments

    def calculate_b2b_diff_features(self, segments):
        # Pre-compute statistics and store in lists
        statistics = { 'mean': [], 'median': [], 'std': [], 'var': [], 'range': [], 
                    'iqr': [], 'skew': [], 'kurtosis': [], 'rms': [], 
                    'samp_entropy': [], 'shannon_entropy': [], 'mean_fd': [], 'std_fd': [] }
        
        for s in segments:
            scaled_data = s * 1e6  # Scale data for precision in kurtosis
            statistics['mean'].append(np.mean(s))
            statistics['median'].append(np.median(s))
            statistics['std'].append(np.std(s))
            statistics['var'].append(np.var(s))
            statistics['range'].append(np.max(s) - np.min(s))
            statistics['iqr'].append(stats.iqr(s))
            statistics['skew'].append(stats.skew(s))
            statistics['kurtosis'].append(stats.kurtosis(scaled_data))
            statistics['rms'].append(np.sqrt(np.mean(np.square(s))))
            statistics['samp_entropy'].append(nolds.sampen(s, 2, 0.2 * np.std(s)))
            statistics['shannon_entropy'].append(entropy(np.histogram(s, density=True)[0], base=2))
            fd = np.diff(s)
            statistics['mean_fd'].append(np.mean(fd))
            statistics['std_fd'].append(np.std(fd))

        # Calculate IQRs of the computed statistics
        return { f'IQR_{key}': stats.iqr(stat) for key, stat in statistics.items() }

    def parallel_feature_computation(self, segments):
        return self.calculate_b2b_diff_features(segments)
    
    def compute_b2b_diff_features(self, data):
        results = {key: [] for key in ['IQR_mean', 'IQR_median', 'IQR_std', 'IQR_var', 'IQR_iqr', 'IQR_range', 
                                    'IQR_skew', 'IQR_kurtosis', 'IQR_rms', 'IQR_samp_entropy', 
                                    'IQR_shannon_entropy', 'IQR_mean_fd', 'IQR_std_fd']}
        
        with ThreadPoolExecutor() as executor:
            # Create a list of future objects
            futures = [executor.submit(self.extract_b2b_segments, ep) for ep in data]
            segments_list = [f.result() for f in tqdm(futures, total=len(futures))]
            
            # Process each segment list in parallel
            feature_futures = [executor.submit(self.parallel_feature_computation, segments) for segments in segments_list]
            for future in tqdm(feature_futures, total=len(feature_futures)):
                feature_result = future.result()
                for key in results:
                    results[key].append(feature_result[key])

        return {key: np.array(val) for key, val in results.items()}

    # def compute_b2b_diff_features(self, data):
    #     results = { key: [] for key in ['IQR_mean', 'IQR_median', 'IQR_std', 'IQR_var', 'IQR_iqr', 'IQR_range', 
    #                                     'IQR_skew', 'IQR_kurtosis', 'IQR_rms', 'IQR_samp_entropy', 
    #                                     'IQR_shannon_entropy', 'IQR_mean_fd', 'IQR_std_fd'] }

    #     for segments in tqdm([self.extract_b2b_segments(ep) for ep in data]):
    #         feature_result = self.calculate_b2b_diff_features(segments)
    #         for key in results:
    #             results[key].append(feature_result[key])

    #     return { key: np.array(val) for key, val in results.items() }