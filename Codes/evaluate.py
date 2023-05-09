import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def get_results_array(directory):
    mious = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path):
            mious.append(np.load(path))
    return np.array(mious)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Script runs evaluation codes on\
                                     active learning results obtained from running\
                                     train.py')
    
    parser.add_argument('--methods', type=str, nargs='+', help='List of Methods\
                        to generate active learning curves and plots for.')
    
    args = parser.parse_args()
    
    # generate training curves
    fig, ax = plt.subplots(figsize=(8,6))
    
    #list of valid sampling methods
    valid_samplings = ['reconstruction', 'entropy', 'random', 'interval']
    for sampling_method in list(args.methods):
        if sampling_method not in valid_samplings:
            raise ValueError('Invalid sampling method specified. Please specify one of "reconstruction",\
                             "entropy", "random", and "interval".')
        
        results_directory = '../Results/'+sampling_method+'/train'
        mious = get_results_array(results_directory)
        ax.plot(np.arange(mious.shape[1]), mious.mean(axis=0), label=sampling_method)
        
        # plot confidence intervals
        mious_upper = mious.mean(axis=0) + 0.9*(mious.std(axis=0)/np.sqrt(mious.shape[0]))
        mious_lower = mious.mean(axis=0) - 0.9*(mious.std(axis=0)/np.sqrt(mious.shape[0]))      
        ax.fill_between(np.arange(mious.shape[1]),  mious_upper, mious_lower, alpha=0.5)
        
    plt.xticks(np.arange(mious.shape[1]), labels=[str(i+1) for i in range(mious.shape[1])])
    plt.xlabel('Cycle', fontsize=10)
    plt.ylabel('mIOU', fontsize=10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('../Figures/miou_train.png')
    
    # generate test curves
    fig, ax = plt.subplots(figsize=(8,6))
    
    #list of valid sampling methods
    valid_samplings = ['reconstruction', 'entropy', 'random', 'interval']
    for sampling_method in list(args.methods):
        if sampling_method not in valid_samplings:
            raise ValueError('Invalid sampling method specified. Please specify one of "reconstruction",\
                             "entropy", "random", and "interval".')
        
        results_directory = '../Results/'+sampling_method+'/test'
        mious = get_results_array(results_directory)
        
        #plot mean
        ax.plot(np.arange(mious.shape[1]), mious.mean(axis=0), label=sampling_method)
        
        # plot confidence intervals
        mious_upper = mious.mean(axis=0) + 0.9*(mious.std(axis=0)/np.sqrt(mious.shape[0]))
        mious_lower = mious.mean(axis=0) - 0.9*(mious.std(axis=0)/np.sqrt(mious.shape[0]))      
        ax.fill_between(np.arange(mious.shape[1]),  mious_upper, mious_lower, alpha=0.5)
        
    plt.xticks(np.arange(mious.shape[1]), labels=[str(i+1) for i in range(mious.shape[1])])
    plt.xlabel('Cycle', fontsize=10)
    plt.ylabel('mIOU', fontsize=10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('../Figures/miou_test.png')
        
    

    
    
    
    