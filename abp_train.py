from ExtractFeatures import ExtractFeatures
import numpy as np
from Utilities import * 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    log_info("Starting script...")
    # Path definitions
    train_features_path = 'data/train_features_10sec.npy' 
    test_features_path = 'data/test_features_10sec.npy'
    train_data_path = 'data/ABP_train_samples_10sec.csv'
    test_data_path = 'data/ABP_test_samples_10sec.csv'

    # Features extraction
    log_info("Extracting features from data")
    train_features = extract_features(train_data_path, 'train', train_features_path)
    test_features = extract_features(test_data_path, 'test', test_features_path)
    X_train, y_train = train_features[:,:-1], train_features[:,-1]
    X_test, y_test = test_features[:,:-1], test_features[:,-1]

    log_info("Standardizing the features (z-scoring)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_and_eval_SVM(X_train, y_train, X_test, y_test)
    train_and_eval_KNN(X_train, y_train, X_test, y_test)
    train_and_eval_DT(X_train, y_train, X_test, y_test)


def train_and_eval_SVM(X_train, y_train, X_test, y_test):
    log_info("Training with SVM")
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)

    log_info("Evaluating the SVM classifier")
    y_pred_train = svm_classifier.predict(X_train)
    log_info(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}\n{classification_report(y_train, y_pred_train)}\n{confusion_matrix(y_train, y_pred_train)}")
    
    y_pred_test = svm_classifier.predict(X_test)
    log_info(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}\n{classification_report(y_test, y_pred_test)}\n{confusion_matrix(y_test, y_pred_test)}")

    log_info(f"Saving the trained SVM model")
    save_model(svm_classifier, 'models/svm_classifier_afib.pkl')


def train_and_eval_KNN(X_train, y_train, X_test, y_test, n_neighbors=5):
    log_info("Training with KNN")
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)

    log_info("Evaluating the KNN classifier")
    y_pred_train = knn_classifier.predict(X_train)
    log_info(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}")
    log_info(f"{classification_report(y_train, y_pred_train)}")
    log_info(f"{confusion_matrix(y_train, y_pred_train)}")

    y_pred_test = knn_classifier.predict(X_test)
    log_info(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
    log_info(f"{classification_report(y_test, y_pred_test)}")
    log_info(f"{confusion_matrix(y_test, y_pred_test)}")

    log_info("Saving the trained KNN model")
    save_model(knn_classifier, 'models/knn_classifier_afib.pkl')


def train_and_eval_DT(X_train, y_train, X_test, y_test, max_depth=None, criterion='gini'):
    log_info("Training with Decision Tree")
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    dt_classifier.fit(X_train, y_train)

    log_info("Evaluating the Decision Tree classifier")
    y_pred_train = dt_classifier.predict(X_train)
    log_info(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}")
    log_info(f"{classification_report(y_train, y_pred_train)}")
    log_info(f"{confusion_matrix(y_train, y_pred_train)}")

    y_pred_test = dt_classifier.predict(X_test)
    log_info(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
    log_info(f"{classification_report(y_test, y_pred_test)}")
    log_info(f"{confusion_matrix(y_test, y_pred_test)}")

    log_info("Saving the trained Decision Tree model")
    save_model(dt_classifier, 'models/dt_classifier_afib.pkl')



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

        print(features.shape, y_clean.shape)
        features_and_labels = np.column_stack((features, y_clean))

        np.save(extraction_save_path, features_and_labels)
        return features_and_labels


def load_data_from_csv(filepath):
    # Load the data from CSV file
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)  
    # Separate features and labels
    X = data[:, :-1]  
    y = data[:, -1]  
    return X, y

if __name__== "__main__":
    main()