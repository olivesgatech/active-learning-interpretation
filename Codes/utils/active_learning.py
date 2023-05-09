# script defines class and associated functions for the active learning training, inference,
# and sampling processes 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataloader import SectionLoader
from utils.network import FaciesSegNet
from numpy.random import choice
from sklearn.metrics import jaccard_score
    

def setup_loader(inds, seismic, labels):
    """Function initializes dataloaders for training and testing"""
    inds = np.array(inds, dtype=int)  # form a numpy array of indices of labeled sections
    dataset = SectionLoader(seismic, labels, sample_inds=inds)
    loader = DataLoader(dataset, batch_size=1)
    return loader 

class ActiveLearningMeter:
    def __init__(self, train_seismic, \
                 test_seismic, train_labels, \
                 test_labels, sampling_method='reconstruction', cycles=1, initial_training_sample=1):
        
        #list of valid sampling methods
        valid_samplings = ['reconstruction', 'entropy', 'random', 'interval']
        if sampling_method not in valid_samplings:
            raise ValueError('Invalid sampling method specified. Please specify one of "reconstruction",\
                             "entropy", "random", and "interval".')
        
        self.cycles = cycles  # no of active learning cycles
        self.sampling_method = sampling_method  # sampling strategy - random or highest error?
        self.reconstruction_errors = []  # stores reconstruction errors over the complete volume at each cycle
        self.entropies = [] # stores entropies over the complete volume at each cycle
        self.m_ious = [[],[]]  # list to store mean IOU for training and test sets after every cycle
        self.training_samples = [initial_training_sample]  # initial traininig inline
        self.training_loader = setup_loader(self.training_samples, train_seismic, train_labels)
        self.test_loader = setup_loader(list(np.arange(0, test_seismic.shape[1])), test_seismic, test_labels)
        self.train_seismic = train_seismic
        self.train_labels = train_labels
        self.test_seismic = test_seismic
        self.test_labels = test_labels
        
        
    def sample(self):        
        if self.sampling_method == 'reconstruction':  # if sampling method samples inline with the highest reconstruction error
            error_profile = self.reconstruction_errors[-1]  # grab the last error profile
            idx = np.array(error_profile.argmax()).reshape(-1,1).astype(int)  # grab index of section with the highest error
            
            # if this idx is already present in the training data, then grab the next highest section
            if list(idx) in self.training_samples:
                sorted_inds = np.argsort(error_profile)
                idx = np.array(sorted_inds[-1]).reshape(-1,1).astype(int)
                
            self.training_samples = self.training_samples + list(idx)
            self.training_loader = setup_loader(self.training_samples, self.train_seismic, self.train_labels)
            return idx
        
        elif self.sampling_method == 'entropy': # random sampling otherwise!
            error_profile = self.entropies[-1]  # grab the last error profile
            idx = np.array(error_profile.argmax()).reshape(-1,1).astype(int)  # grab index of section with the highest error
            
            # if this idx is already present in the training data, then grab the next highest section
            if list(idx) in self.training_samples:
                sorted_inds = np.argsort(error_profile)
                idx = np.array(sorted_inds[-1]).reshape(-1,1).astype(int)
                
            self.training_samples = self.training_samples + list(idx)
            self.training_loader = setup_loader(self.training_samples, self.train_seismic, self.train_labels)
            return idx
        
        elif self.sampling_method == 'random': # random sampling otherwise!
            idx_nums = np.arange(self.train_seismic.shape[1])
            sampled_idx = choice(idx_nums, 1, replace=True).astype(int)
            self.training_samples = self.training_samples + list(sampled_idx)
            self.training_loader = setup_loader(self.training_samples, self.train_seismic, self.train_labels)
            return sampled_idx
        
        elif self.sampling_method == 'interval':
            new_sample = [self.training_samples[-1] + 50]  # grab the section 50 samples away from the last one
            self.training_samples = self.training_samples + new_sample
            self.training_loader = setup_loader(self.training_samples, self.train_seismic, self.train_labels)
            return np.array(new_sample[0]).astype(int)
        
        
    def update_error_profile(self,error_profile):
        if self.sampling_method == 'reconstruction':
            self.reconstruction_errors.append(error_profile)
        else:
            self.entropies.append(error_profile)
    
    def update_iou(self,iou_train, iou_test):
        self.m_ious[0].append(iou_train)
        self.m_ious[1].append(iou_test)


def run_active_learning(train_seismic, \
                        test_seismic, train_labels, \
                        test_labels, sampling_method='reconstruction', cycles=1, initial_training_sample=1):
    """Function runs the specified number of Active Learning cycles on the data, validating performance on the given validation
    data after each cycle."""
    
    
    active_learning_meter = ActiveLearningMeter(cycles=cycles, initial_training_sample=initial_training_sample, \
                                                train_seismic=train_seismic, test_seismic=test_seismic, \
                                                train_labels=train_labels, test_labels=test_labels, \
                                                sampling_method=sampling_method)

    # initializing model outside of loop --- initialize weigths from last cycle each cycles
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model = FaciesSegNet(n_class=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    criterion1 = nn.CrossEntropyLoss() # Set up criterion
    criterion2 = nn.MSELoss()
    
    num_epochs = 300

    for cycle in range(cycles):
        
        train_loader = active_learning_meter.training_loader
        test_loader = active_learning_meter.test_loader
        
        # train model
        for epoch in range(num_epochs):
            for iteration, (section, label) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                pred_sec, reconstruct = model(section)
                segmentation_loss = criterion1(pred_sec, label)
                recons_loss = criterion2(reconstruct, section)
                loss = segmentation_loss + recons_loss
                loss.backward()
                optimizer.step()         
                
                            
        # perform inference on train/unlabeled pool
        uncertainty_scores = []
        pred_segmentation_vol = np.zeros(train_seismic.shape)
        
        model.eval()
        with torch.no_grad():
            for i in range(train_seismic.shape[1]):
                
                section = torch.from_numpy(train_seismic[:,i,:].T).to(device).type(torch.float).unsqueeze(0).unsqueeze(0)
                label = torch.from_numpy(train_labels[:,i,:].T).to(device).type(torch.long).unsqueeze(0)
                
                out, recons = model(section)
                
                if sampling_method=='entropy':
                    # compute entropy for section and add to saved result
                    out_softmax = torch.nn.functional.softmax(out,dim=1)
                    entropy = - torch.sum((out_softmax) * torch.log(out_softmax + 1e-2))
                    uncertainty_scores.append(entropy.item())
                    
                else:
                    uncertainty_scores.append(criterion2(recons,section).item())
                

                # predict train volume
                pred_segmentation_vol[:,i,:] = out.argmax(1).detach().cpu().numpy().squeeze().T
        
        # perform inference on test
        pred_test_vol = np.zeros(test_seismic.shape)
        model.eval()
        with torch.no_grad():
            for i, (section, label) in enumerate(test_loader):
                out, _ = model(section)
                pred_test_vol[:, i,:] = out.argmax(1).detach().cpu().numpy().squeeze().T
        
        
        IOU_train = jaccard_score(pred_segmentation_vol.flatten(), train_labels.flatten(), labels=list(range(6)), average=None)
        IOU_test = jaccard_score(pred_test_vol.flatten(), test_labels.flatten(), labels=list(range(6)), average=None)
        print('Cycle No: {} | Mean Train IOU: {:0.4f} | Mean Test IOU : {:0.4f}'.format(cycle, IOU_train.mean(), IOU_test.mean()))
        
        
        active_learning_meter.update_error_profile(np.array(uncertainty_scores))
        active_learning_meter.update_iou(IOU_train.mean(), IOU_test.mean())
        
        
        if cycle < cycles-1:
            sampled_inlines = active_learning_meter.sample()            
            print("Sampled Inlines: ", sampled_inlines,'\n\n')
        else:
            pass

        
        
    return model, active_learning_meter   