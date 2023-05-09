# This script takes in arguments from command line to run active learning

import argparse
import os
import matplotlib.pyplot as plt
from utils.data_preprocess import load_data, standardize_features, train_test_split
from utils.active_learning import run_active_learning


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script runs active learning\
                                     using settings provided in command line')
    parser.add_argument('--path_seismic', type=str, help='Path to seismic volume in numpy format.')
    parser.add_argument('--path_labels', type=str, help='Path to labels volume in numpy format.')
    parser.add_argument('--training_inds', type=int, nargs='+', help='tuple of \
                        integers specifying the first and last indices of the \
                        training split in the seismic and label volumes.')
    parser.add_argument('--sampling_method', type=str, help='Active learning \
                        query strategy (default: reconstruction)', default='reconstruction')
    parser.add_argument('--cycles', type=int, help='Number of cycles to run active learning for.'\
                        , default=5)
    parser.add_argument('--trials', type=int, help='Number of active learning trials with\
                        the selected sampling strategy', default=1)
    

    args = parser.parse_args()

    # load seismic and label volumes, standardize, and do train-test splitting
    seismic, labels = load_data(args.path_seismic, args.path_labels)
    seismic_normalized = standardize_features(seismic)  # normalize seismic
    train_seismic, test_seismic,\
    train_labels, test_labels = train_test_split(seismic_normalized, labels, tuple(args.training_inds))
    
    # run active learning for required number of trials
    for trial_num in range(args.trials):
        model,  active_learning_meter = run_active_learning(cycles=args.cycles,\
                                                            initial_training_sample=1,\
                                                            train_seismic,\
                                                            test_seismic,\
                                                            train_labels,\
                                                            test_labels,\
                                                            sampling_method=args.sampling_method)
        
        # save model and mious in relevant directories

        # check if save directory exists for models
        model_save_directory = 'models/'+args.sampling_method
        if not os.path.exists(model_save_directory):
            os.makedirs(model_save_directory)
          
        # save model    
        torch.save(model.state_dict(),os.path.join(model_save_directory, \
                   'model_'+args.sampling_method+'_trial_'+str(trial)+'.pth'))
        
        # check if save directory exists for training mious
        train_mious_save_directory = 'results/'+args.sampling_method
        if not os.path.exists(train_mious_save_directory):
            os.makedirs(train_mious_save_directory)
            
        # save training mious
        np.save(os.path.join(train_mious_save_directory,\
                             'train_'+args.sampling_method+'_trial_'+\
                             str(trial)+'.npy'), active_learning_meter.mious[0])
        
        # check if save directory exists for test mious
        test_mious_save_directory = 'results/'+args.sampling_method
        if not os.path.exists(test_mious_save_directory):
            os.makedirs(test_mious_save_directory)
            
        # save test mious
        np.save(os.path.join(test_mious_save_directory,\
                             'test_'+args.sampling_method+'_trial_'+\
                             str(trial)+'.npy'), active_learning_meter.mious[1])
        
        
    
    
    
    
    