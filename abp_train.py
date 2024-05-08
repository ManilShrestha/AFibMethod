from ExtractFeatures import ExtractFeatures
import numpy as np
from Utilities import * 
import os

def main():
    log_info("Starting script...")
    
    train_features_path = 'data/train_features.npy' 
    test_features_path = 'data/test_features.npy'

    train_data_path = 'data/ABP_train_samples.csv'
    test_data_path = '../data/ABP_test_samples.csv'

    train_features = extract_features(train_data_path, 'train', train_features_path)
    test_features = extract_features(test_data_path, 'train', test_features_path)

    X_train, y_train = train_features[:,:-1], train_features[:,-1]
    X_test, y_test = test_features[:,:-1], test_features[:,-1]



def extract_features(data_filepath, feature_type, extraction_save_path):
    """Checks and extracts features for the give filepath and feature type

    Args:
        data_filepath (str): filepath to the datafile with raw signal data
        feature_type (str): either 'train' or 'test'
        extraction_save_path (str): path where extracted features are saved
    """
    if os.path.exists(extraction_save_path):
        log_info(f"{feature_type} features already computed")
        return np.load(extraction_save_path)
    else:
        log_info(f"Loading {feature_type} data from csv: {data_filepath}")
        X, y = load_data_from_csv(data_filepath)
        X_clean, y_clean = clean_dead_signals(X, y)
        log_info("Extracting features")
        ef = ExtractFeatures(X_clean)
        features  = ef.get_features()
        features_and_labels = np.vstack((features, y_clean))

        np.save(extraction_save_path, features_and_labels)
        return features_and_labels


def load_data_from_csv(filepath):
    # Load the data from CSV file
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)  # skip_header is used if your CSV has a header row
    # Separate features and labels
    X = data[:, :-1]  # all rows, all columns except the last
    y = data[:, -1]   # all rows, last column
    return X, y

if __name__== "__main__":
    main()