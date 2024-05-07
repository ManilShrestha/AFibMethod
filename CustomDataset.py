import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.signal import resample_poly
import torch
import pandas as pd


class CustomHD5Dataset(Dataset):
	"""
	This class has methods to read the hd5 file and dump the data into csv according to segment provided in csv_file_path.
	Since reading hd5 file is too slow, dumping data into csv removes this bottleneck.
	"""
	def __init__(self, hd5_path, csv_file_path, sampling_freq=125, sample_len_sec=3, original_freq=125, data_group_name='Waveforms/ART_na', timestamp_group_name='Waveforms/ART_na_Timestamps' ):
		"""
		
		Args:
			hd5_path:        Filepath to the hdf5 file
			csv_file_path:   This csv contains each interval to consider
		
		"""
		
		self.csv_file_path = csv_file_path
		self.hd5_path = hd5_path
		self.sample_len = sample_len_sec * sampling_freq
		self.sampling_freq = sampling_freq
		self.sample_len_sec = sample_len_sec
		
		with h5py.File(hd5_path, 'r') as file:
			dataset = file[data_group_name]
			timestamp = file[timestamp_group_name]
	
			data = dataset[:]
			timestamp = timestamp[:]
		
		self.intervals = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)
		
		if sampling_freq!=original_freq:
			down_val = int(original_freq/sampling_freq)
			# Resampling the data for efficiency
			self.data = resample_poly(data, up=1, down=down_val)
			self.timestamp = timestamp[::down_val]
		else:
			self.data = data
			self.timestamp = timestamp

		
		
		
	def __len__(self):
		return len(self.intervals)
	
	def __getitem__(self, idx):
		
		start_time, end_time, label = self.intervals[idx]
		
		# Efficiently find the start and end indices (assuming sorted timestamps)
		# This is a placeholder for a more efficient search method
		
		################ BOTTLE NECK #####################
		start_idx = np.searchsorted(self.timestamp, start_time, side='left')
		end_idx = start_idx + (self.sampling_freq *  self.sample_len_sec)
		# For 125Hz, the sample_len is 1250 for 10sec sample
		##################################################

		interval_data = self.data[start_idx:end_idx]

		# Define a fixed sequence length
		fixed_length = self.sample_len  # Example fixed length

		# Pad or truncate the sequence to the fixed length
		if len(interval_data) > fixed_length:
			interval_data = interval_data[:fixed_length]  # Truncate

		elif len(interval_data) < fixed_length:
			padding = fixed_length - len(interval_data)  # Calculate padding size
			interval_data = np.pad(interval_data, (0, padding), 'constant', constant_values=0)  # Pad

		# Convert to tensor
		interval_data = torch.tensor(interval_data, dtype=torch.float)

		return idx, interval_data, label
	


class CSVSignalDataset(Dataset):
	def __init__(self, csv_data_file, train_data_file):
		"""
		Args:
			csv_data_file: Path to the CSV file containing signals.
			train_data_file: To get the mean and std to standardize
		"""
		  
		self.csv_data_file = csv_data_file
		self.train_data_file = train_data_file

		self.data_frame = pd.read_csv(csv_data_file, skiprows=1)
		self.labels = torch.tensor(self.data_frame.iloc[:, 1].values).long()
		self.data = torch.tensor(self.data_frame.iloc[:, 2:].values).float()
		self.mean, self.std = self.get_mean_std()
		
	
	def get_mean_std(self):
		#################
		csv_path = self.train_data_file
		data = np.loadtxt(csv_path, delimiter=',',skiprows=1)[:,2:]
		
		# Calculate the mean and standard deviation
		mean = np.mean(data, axis=0)
		std = np.std(data, axis=0)

		# Ensure std is not zero to avoid division by zero
		std = np.where(std == 0, 1, std)
		return mean, std

	def __len__(self):
		# Return the number of rows in the DataFrame
		return len(self.data_frame)
	
	def __getitem__(self, idx):
		"""
		Args:
			idx: Index of the data sample.
			
		Returns:
			A tuple (ecg_sample, label) where ecg_sample is the ECG data as a tensor
			and label is the corresponding label as a tensor.
		"""
		sample = (self.data[idx] - self.mean) / self.std
		
		label = self.labels[idx]
		return idx, sample, label