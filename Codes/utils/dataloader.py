# Create a custom dataset class for loading seismic sections into the network for training

from torch.utils.data import Dataset
import torch


class SectionLoader(Dataset):
    """Dataset class for loading F3 patches and labels"""

    def __init__(self, seismic_cube, label_cube, sample_inds):
        """Initializer function for the dataset class

        Parameters
        ----------
        seismic_cube: array_like
                    3D ndarray of floats representing seismic amplitudes

        label_cube: array_like
                 3D ndarray same dimensions as seismic_cube containing label information.
                 Each value is [0,num_classes]
                    
        """

        self.seismic = seismic_cube
        self.label = label_cube
        self.indices = sample_inds
        

    def __getitem__(self, index):
        """Obtains the image crops relating to each section in the given orientation.

        Parameters
        ----------
        index: int
             Integer specifies the section number along the given orientation.

        Returns
        -------
        images: ndarray of shape (1, H, W)
              Returns inline section specified by index"""

        section_num = self.indices[index]        

        section = self.seismic[:, section_num, :]
        label_section = self.label[:, section_num, :]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        section = torch.from_numpy(section).to(device).type(torch.float).unsqueeze(0)
        label_section = torch.from_numpy(label_section).to(device).type(torch.long)
                  
        return section, label_section
        
    
    def __len__(self):
        """Retrieves total number of training samples"""
        return self.inds.size
