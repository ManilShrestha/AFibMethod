import h5py
import pandas as pd
import numpy as np
from Utilities import *
from tqdm import tqdm
from CustomDataset import CustomHD5Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class Preprocess:
	def __init__(self, hd5file_path, annotation_path, annotation_metadata, data_group_name='Waveforms/ART_na', timestamp_group_name='Waveforms/ART_na_Timestamps', segment_length_sec=3, sampling_frequency=125, signal_type="ABP"):
		"""
		hd5file_path: Path to the hd5 file with data
		annotation_path: Path to the annotation file (csv)
		annotation_metadata: This dict will have information about the annotation file, which column to use etc. Information on 'modality' and 'location' columns.
		data_group_name: The group in hdf5 file with data in it
		timestamp_group_name: The group in hdf5 file with timestamp data in it
		segment_length_sec: The length of each segment of the signal to consider
		"""
		
		self.hd5file_path=hd5file_path
		self.annotation_path=annotation_path
		self.data_group_name=data_group_name
		self.timestamp_group_name = timestamp_group_name
		self.segment_length_sec = segment_length_sec
		self.sampling_frequency = sampling_frequency
		self.signal_type = signal_type
		
		df_annotation = pd.read_csv(annotation_path)
		df_annotation_filtered = df_annotation[(df_annotation['modality']==annotation_metadata['modality']) & (df_annotation['location']==annotation_metadata['location'])]
		self.artifacts = df_annotation_filtered[["start_time","end_time"]].to_numpy() * int(annotation_metadata['scale_wrt_hd5'])		 
	
		self.data, self.timestamp = self.read_hd5()
		self.test_instances = []


	def read_hd5(self):
		"""
		This method reads the hd5 file and returns the data and timestamp group of the file
		"""
		with h5py.File(self.hd5file_path, 'r') as file:
			dataset = file[self.data_group_name]
			timestamp = file[self.timestamp_group_name]

			return dataset[:], timestamp[:]

	
	def create_train_val_test_set(self, train_data_filepath, val_data_filepath, test_data_filepath):
		"""1. First get the total number of intervals with artifact
		2. Balance by randomly sampling with no overlap to the artifact as negative cases
		3. Do 6:3:1 split for train:test:validation 

		Args:
			train_data_filepath (Str): Path to store training data
			val_data_filepath (_type_): Path to store validation data
			test_data_filepath (_type_): Path to store test data
		"""
		log_info('Fetching data...')
		artifact_raw = self.get_artifact_data()
		non_artifact_raw = self.get_non_artifact_data(len(artifact_raw))

		artifact_raw = np.array(artifact_raw)
		non_artifact_raw = np.array(non_artifact_raw)

		log_info(f"There are total of {len(artifact_raw)} positive samples and {len(non_artifact_raw)} negative samples. And there are {artifact_raw.shape[1]} columns.")

		# Create label columns
		artifact_labels = np.ones((artifact_raw.shape[0], 1))  
		non_artifact_labels = np.zeros((non_artifact_raw.shape[0], 1))  

		# Concatenate the label columns to the original arrays
		artifact_labeled = np.hstack((artifact_raw, artifact_labels)) 
		non_artifact_labeled = np.hstack((non_artifact_raw, non_artifact_labels)) 
		
		combined_dataset = np.vstack((artifact_labeled, non_artifact_labeled))

		# Separate features and labels
		X = combined_dataset[:, :-1]  # all rows, all columns except the last
		y = combined_dataset[:, -1]   # all rows, only the last column

		# First split to separate out the test set
		X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

		# Second split to divide the remaining data into train and validation sets
		X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, stratify=y_temp, random_state=42)

		# Combine features and labels back for each set
		train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
		val_data = np.hstack((X_val, y_val.reshape(-1, 1)))
		test_data = np.hstack((X_test, y_test.reshape(-1, 1)))

		# Save to CSV files
		np.savetxt(train_data_filepath, train_data, delimiter=",")
		np.savetxt(val_data_filepath, val_data, delimiter=",")
		np.savetxt(test_data_filepath, test_data, delimiter=",")

		# Verify the split
		log_info(f"Train data shape: {train_data.shape}")
		log_info(f"Validation data shape: {val_data.shape}")
		log_info(f"Test data shape: {test_data.shape}")



	def get_artifact_data(self, sliding_step_sec=0.5):
		"""Get artifact data based on annotation file with sliding window."""
		segment_length = self.segment_length_sec * self.sampling_frequency
		sliding_step = int(sliding_step_sec * self.sampling_frequency) 

		artifact_raw = []
		for art in self.artifacts:
			# Extend indices to include data from before and after the artifact
			extended_start_idx = np.searchsorted(self.timestamp, art[0], side='left') - segment_length
			extended_start_idx = max(extended_start_idx, 0)  # Ensure it does not go below zero

			extended_end_idx = np.searchsorted(self.timestamp, art[1], side='left') + segment_length
			extended_end_idx = min(extended_end_idx, len(self.data))  # Ensure it does not go beyond data length

			# Extract the extended data
			# extended_data = self.data[extended_start_idx:extended_end_idx]

			# Implement sliding window within the extended data
			for start in range(extended_start_idx, extended_end_idx-segment_length, sliding_step):
				end = start + segment_length
				
				window_data = self.data[start:end]
				window_ts = self.timestamp[start:end]

				if has_artifact(window_ts, self.artifacts):
					# Check if more than 80% of the data in the window is less than zero
					if np.sum(window_data <= 0) / len(window_data) <= 0.8:
						artifact_raw.append(window_data)

		return artifact_raw
		

	def get_non_artifact_data(self, num_positive_samples):
		"""Get non-artifact data. Randomly select the segment length worth of data
		   and then find if it has artifact, if not append.
		"""
		segment_length = self.segment_length_sec * self.sampling_frequency


		# Randomly get a segment that is of length given as segment_length_sec*sampling_frequency
		# If has artifact, then append to artifact list else append to non-artifact list

		# reduced_range = int(len(self.timestamp)/segment_length)

		# Generate num_positive_samples*2 unique random values from 0 to 58360000 without replacement
		random_values = np.random.choice(len(self.timestamp), num_positive_samples*2, replace=False)

		count_negative, i = 0, 0

		non_artifact_raw=[]
		while count_negative<num_positive_samples:
			start_idx = random_values[i]
			temp_ts = self.timestamp[start_idx : start_idx+segment_length]
			if not has_artifact(temp_ts, self.artifacts):
				temp_data = self.data[start_idx: start_idx+segment_length]
				if len(temp_data)==segment_length:
					non_artifact_raw.append(self.data[start_idx: start_idx+segment_length])
					count_negative+=1
			i+=1

		return non_artifact_raw