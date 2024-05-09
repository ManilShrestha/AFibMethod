from datetime import datetime
import numpy as np
import pickle

def log_info(log_message):
	print( datetime.now().strftime("%H:%M:%S"),":\t ", log_message , "\n")


def has_artifact(candidate_interval, artifacts):
		for artifact in artifacts:
			# Calculate the maximum start time and minimum end time between candidate_interval and artifact
			start_max = max(candidate_interval[0], artifact[0])
			end_min = min(candidate_interval[1], artifact[1])
			
			# Check for overlap
			if start_max < end_min:
				# If there is an overlap, return True
				return True
		
		# If no overlap is found with any artifact, return False
		return False

def has_overlap(candidate_interval, test_instances):
    
    test_set = np.array(test_instances)[:,0:2]
    
    for sample in test_set:
        # Calculate the maximum start time and minimum end time between candidate_interval and artifact
        start_max = max(candidate_interval[0], sample[0])
        end_min = min(candidate_interval[1], sample[1])
        
        # Check for overlap
        if start_max < end_min:
            # If there is an overlap, return True
            return True
    
    # If no overlap is found with any artifact, return False
    return False

	
def clean_dead_signals(X, y):
    # Removing values from X that are less than or equal to 0 for more than 50% of duration
    mask = X <= 0
    count_le_zero = np.sum(mask, axis=1)
    # Find the rows where more than 50% of the elements are <= 0
    rows_to_remove = count_le_zero > (X.shape[1] / 2)
    rows_to_keep = ~rows_to_remove

    # Apply the mask to filter out the rows in
    return X[rows_to_keep], y[rows_to_keep] 


def save_model(model, path):
    with open(path,'wb') as f:
        pickle.dump(model,f)