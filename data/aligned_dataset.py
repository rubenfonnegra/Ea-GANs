import os.path
import random
import torchvision.transforms as T
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from skimage import io
import SimpleITK as sitk
import numpy as np
import pandas as pd



##
## Custom scaling transform
## 
class min_max_scaling (torch.nn.Module):
    """Scales a tensor image within [range.min, range.max]
    This transform does not support PIL Image.
    Given range: ``range[0], range[1]`` this transform will scale each input
    ``torch.*Tensor`` i.e.,
    ``output = (range.max - range.min) * ( (input - input.min ) / (tensor.max - tensor.min) ) - range.min ``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        range (sequence): Sequence of min-max values to scale.
        inplace(bool,optional): Bool to make this operation in-place.
    """
    
    def __init__(self, in_range = None, out_range = [-1,1], inplace=False):
        """"""
        super().__init__()
        self.inplace = inplace
        self.out_range = np.asarray(out_range)
        self.in_range  = np.asarray(in_range) if in_range else None
    
    def forward(self, tensor: torch.Tensor, *kwargs) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be scaled.
        Returns:
            Tensor: Scaled Tensor image.
        """
        # in_range = self.in_range if isinstance(self.in_range, np.ndarray) else np.asarray([tensor.min(), tensor.max()]) 
        in_range = self.in_range if isinstance(self.in_range, np.ndarray) else np.asarray([tensor.min().detach().cpu().numpy(), tensor.max().detach().cpu().numpy()]) 
        #return F.normalize(tensor, self.mean, self.std, self.inplace)
        return (self.out_range.max()-self.out_range.min())*(tensor-in_range.min()) / (in_range.max()-in_range.min()) + self.out_range.min()
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(:\n\t(in_range min={self.in_range.min()}, in_range max={self.in_range.max()}), \
                                            \n\t(out_range min={self.out_range.min()}, out_range max={self.out_range.max()}))"


class AlignedDataset(BaseDataset):   ## train
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.dir_A = opt.dataroot + opt.inp_seq + "/train_" + opt.quality
        # self.dir_B = opt.dataroot + opt.out_seq + "/train_" + opt.quality
        # # self.dir_AB = os.path.join(opt.dataroot) #, opt.phase
        # # self.AB_paths = sorted(make_dataset(self.dir_AB))
        # self.A_paths = sorted(make_dataset(self.dir_A))
        # self.B_paths = sorted(make_dataset(self.dir_B))

        data = pd.read_csv(opt.dataroot + "/train_" + opt.quality + ".csv")
        
        self.A_paths = data[opt.inp_seq].tolist()
        self.B_paths = data[opt.out_seq].tolist()

        assert(opt.resize_or_crop == 'resize_and_crop')

        # self.transform = torch.from_numpy
        self.transform = T.Compose([T.Resize([opt.loadSize, opt.loadSize]), 
                                    T.ToTensor(), 
                                    min_max_scaling(out_range = [-1,1])])
        self.eps = 10**(-12)

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        # img0 = io.imread(A_path, plugin = 'simpleitk')
        A = Image.open(self.root + A_path).convert("F") #.convert('RGB')
        B = Image.open(self.root + B_path).convert("F") #.convert('RGB')

        # if len(A.size) < 3: 
        #     A = np.array(A)[np.newaxis,:,:]
        #     B = np.array(B)[np.newaxis,:,:]
        #     nchannels = 1
        # else: 
        #     A = np.array(A).transpose(2,0,1)
        #     B = np.array(B).transpose(2,0,1)
        #     nchannels = 3

        # # AB = np.zeros((1,self.opt.fineSize*2,self.opt.fineSize,self.opt.fineSize))
        # AB = np.zeros((1,nchannels*2,self.opt.loadSize,self.opt.loadSize))
        # AB[0,0:nchannels,:,:] = np.array(A)
        # AB[0,nchannels:nchannels*2,:,:] = np.array(B)

        # AB = self.transform(AB)

        # A = AB[0,0:nchannels,:,:]#[None, :]
        # B = AB[0,nchannels:nchannels*2,:,:]#[None, :]
        
        A = self.transform(A)
        B = self.transform(B)

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, #[None, :,:,:], 
                'B': B, #[None, :,:,:],
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'





class TestAlignedDataset(BaseDataset):   
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(opt.resize_or_crop == 'resize_and_crop')



        self.transform = torch.from_numpy

    def __getitem__(self, index):

        AB_path = self.AB_paths[index]
        img0 = io.imread(AB_path, plugin = 'simpleitk')
        img0 = img0.reshape((2*self.opt.fineSize,self.opt.fineSize,self.opt.fineSize))
        AB = np.zeros((1,self.opt.fineSize*2,self.opt.fineSize,self.opt.fineSize))
        AB[0,0:self.opt.fineSize,:,:] = img0[0:self.opt.fineSize,:,:]

        AB[0,self.opt.fineSize:self.opt.fineSize*2,:,:] = img0[self.opt.fineSize:(self.opt.fineSize*2),:,:]


        AB = self.transform(AB)



        A = AB[:,:self.opt.fineSize,:,:]
        B = AB[:,self.opt.fineSize:self.opt.fineSize*2,:,:]



        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(3) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(3, idx)
            B = B.index_select(3, idx)


        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'