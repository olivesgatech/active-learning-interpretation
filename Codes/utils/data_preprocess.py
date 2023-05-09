# script contains functions to preprocess data

import numpy as np

def load_data(path_seismic, path_labels):
    """loads and returns seismic volume and corresponding labels from the paths provided"""
    
    seismic = np.load(path_seismic)
    labels = np.load(path_labels)
    
    return seismic, labels


def standardize_features(seismic):
    """standardizes seismic amplitudes have zero mean and unit variance"""

    seismic_normalized = (seismic - seismic.min()) / (seismic.max() - seismic.min())   
    
    return seismic_normalized


def train_test_split(seismic, labels, train_ind_range):
    """Splits loaded volumes into train and test splits.
    
    Args:
    -----
        seismic: array_like
                3D numpy array containing the seismic volume
        labels: array_like
                3D numpy array containing the labels
        train_ind_range: tuple of ints
                tuple specifying the first and last section to include in the 
                training split. Rest goes into test split.
            
    Returns:
    --------
        train_seismic: array_like
                3D numpy array containing the training split from the seismic
                volume
        test_seismic: array_like
                3D numpy array containing the test split from the seismic
                volume
        train_labels: array_like
                3D array containing training split labels
        test_labels: array_like
                3D array containing test split labels
        """
    
    if len(train_ind_range)!=2:
        raise Exception('Please enter two integers to specify the training\
                        range in the seismic volume.')
        
    elif train_ind_range[0] >= train_ind_range[1]:
        raise Exception('The beginning index should be lower than the last\
                        index!')
    
    elif train_ind_range[0] < 0 or train_ind_range[1] >= seismic.shape[1]:
        raise Exception('One or both of the indices are out of range.')
    
    elif train_ind_range[0]==0:  # if the first index is 0
        train_seismic = seismic[:, train_ind_range[0]:train_ind_range[1], :]
        train_labels = labels[:, train_ind_range[0]:train_ind_range[1], :] 
        test_seismic = seismic[:, train_ind_range[1]:, :]
        test_labels = labels[:, train_ind_range[1]:, :] 
        
    elif train_ind_range[1] == seismic.shape-1: # if the last index is equal to volume dimensionality
        train_seismic = seismic[:, train_ind_range[0]:train_ind_range[1], :]
        train_labels = labels[:, train_ind_range[0]:train_ind_range[1], :] 
        test_seismic = seismic[:, :train_ind_range[0]:, :]
        test_labels = labels[:, :train_ind_range[0]:, :]
        
        # if training inds lie in between the end points
    elif train_ind_range[0] > 0 and train_ind_range[1] < seismic.shape-1:
        train_seismic = seismic[:, train_ind_range[0]:train_ind_range[1], :]
        train_labels = labels[:, train_ind_range[0]:train_ind_range[1], :] 
        test_seismic = seismic[:, np.r_[:train_ind_range[0], train_ind_range[1]:], :]
        test_labels = labels[:, np.r_[:train_ind_range[0], train_ind_range[1]:], :]
        
    return train_seismic, test_seismic, train_labels, test_labels
        
        
        
        
        


