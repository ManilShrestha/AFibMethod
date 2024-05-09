from Preprocess import Preprocess
from Utilities import *


def main():
    log_info("Starting script...")

    hdf5_file_path = '/home/ms5267@drexel.edu/moberg-precicecap/data/Patient_2021-12-21_04_16.h5'
    annotation_file = '/home/ms5267@drexel.edu/moberg-precicecap/data/20240207-annotations-export-workspace=precicecap-patient=7-annotation_group=90.csv'
    annotation_metadata = {
        'modality':'ART'
        ,'location':'na'
        ,'scale_wrt_hd5':1e3
    }
    segment_length_sec = 5

    preprocess_abp = Preprocess(hdf5_file_path,annotation_file, annotation_metadata, segment_length_sec=segment_length_sec)
 
    preprocess_abp.create_train_val_test_set('data/ABP_train_samples_5sec.csv', 'data/ABP_val_samples_5sec.csv', 'data/ABP_test_samples_5sec.csv')

if __name__== "__main__":
    main()