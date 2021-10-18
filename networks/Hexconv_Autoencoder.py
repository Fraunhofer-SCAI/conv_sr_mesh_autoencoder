import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

from hexagdly.hexagdly_py import Conv2d

#first pooling
class first_pooling(nn.Module):
    r"""
    input: 13x13 (kernel size 2)
    output: 7x7 (kernel size 1)
    """

    def __init__(
        self, kernel_size=2, stride=1
    ):
        super(first_pooling, self).__init__()
        
        # has to be like this for now. this corresponds to an refinement of level 3 for the semi regular mesh
        self.height_padded=13
        self.width_padded=13
        
        self.base_vertices_at_refine3 = torch.tensor(np.array([[1,0],[3,0],[5,0],[7,0],[9,0],[11,0],
                            [0,2],[2,2],[4,2],[6,2],[8,2],[10,2],[12,2],
                            [1,4],[3,4],[5,4],[7,4],[9,4],[11,4],
                            [2,6],[4,6],[6,6],[8,6],[10,6],
                            [3,8],[5,8],[7,8],[9,8],
                            [4,10],[6,10],[8,10],
                            [5,12],[7,12]]))
                
        # map the neighborhood average of the vertices base_vertices_at_refine3 to
        # the vertices in smaller tensor at mapto_refine3
        self.basex = self.base_vertices_at_refine3[:,0].long() 
        self.basey = self.base_vertices_at_refine3[:,1].long() 
        # indices of the neighboring vertices, make sure we dont leave the patch
        self.basexminus1 = torch.maximum((self.basex-1),
                                         0*torch.ones(len(self.basex))).long() 
        self.basexplus1 = torch.minimum((self.basex+1),
                                        (self.height_padded-1)*torch.ones(len(self.basex))).long() 
        self.baseyminus1 = torch.maximum((self.basey-1),
                                        0*torch.ones(len(self.basey))).long() 
        self.baseyplus1 = torch.minimum((self.basey+1),
                                        (self.width_padded-1)*torch.ones(len(self.basey))).long() 
        
        self.mapto_refine3 = torch.zeros(self.base_vertices_at_refine3.shape).long() 
        self.mapto_refine3[:,0] = torch.floor(self.base_vertices_at_refine3[:,0]/2)  
        self.mapto_refine3[:,1] = torch.floor((self.base_vertices_at_refine3[:,1])/2)
        self.mapto_refine3[:,0] += ((self.mapto_refine3[:,1])+1)%2
        
        self.kernel_size = kernel_size
        self.stride = stride
        


    def forward(self, input):
        
        batchsize, filters = input.shape[:2]
        
        # if we want to change average to something else, change the following lines
        # work with a flat tensor (tmp). write new values there
        # center vertices
        tmp = input[:,:,self.basex,self.basey]
        # sum vertex values in same column
        tmp += input[:,:,self.basexminus1,self.basey]
        tmp += input[:,:,self.basexplus1,self.basey]
        # sum vertex values in neighboring column
        # because we pick every other column, the kernel is always shifted to the top
        # same row
        tmp += input[:,:,self.basex,self.baseyplus1]
        tmp += input[:,:,self.basex,self.baseyminus1]
        # upper row
        tmp += input[:,:,self.basexminus1,self.baseyplus1]
        tmp += input[:,:,self.basexminus1,self.baseyminus1]
        # average
        tmp  = 1./7 * tmp

        # write the pooling results to the output tensor
        height_padded=7
        width_padded=7
        data_out = torch.zeros((batchsize, filters, height_padded, width_padded), device=input.device)
        data_out[:,:,self.mapto_refine3[:,0],self.mapto_refine3[:,1]] = tmp

        return data_out

#second pooling
class second_pooling(nn.Module):
    r"""
    input: 7x7 (kernel size 1)
    output: 3x3 (kernel size 0)
    """

    def __init__(
        self, kernel_size=1, stride=1
    ):
        super(second_pooling, self).__init__()
        
        # has to be like this for now. this corresponds to an refinement of level 3 for the semi regular mesh
        self.height_padded=7
        self.width_padded=7
        
        self.base_vertices_at_refine2 = torch.tensor(np.array([[1,1],[3,1],[5,1],
                                                        [2,3],[4,3],
                                                        [3,5]]))
                
        # map the neighborhood average of the vertices base_vertices_at_refine3 to
        # the vertices in smaller tensor at mapto_refine3
        self.basex = self.base_vertices_at_refine2[:,0].long() 
        self.basey = self.base_vertices_at_refine2[:,1].long() 
        # indices of the neighboring vertices, make sure we dont leave the patch
        self.basexminus1 = torch.maximum((self.basex-1),
                                         0*torch.ones(len(self.basex))).long() 
        self.basexplus1 = torch.minimum((self.basex+1),
                                        (self.height_padded-1)*torch.ones(len(self.basex))).long() 
        self.baseyminus1 = torch.maximum((self.basey-1),
                                        0*torch.ones(len(self.basey))).long() 
        self.baseyplus1 = torch.minimum((self.basey+1),
                                        (self.width_padded-1)*torch.ones(len(self.basey))).long() 
        
        self.mapto_refine2 = torch.zeros(self.base_vertices_at_refine2.shape).long() 
        self.mapto_refine2[:,0] = torch.floor(self.base_vertices_at_refine2[:,0]/2)  
        self.mapto_refine2[:,1] = torch.floor((self.base_vertices_at_refine2[:,1])/2)
        self.mapto_refine2[:,0] -= ((self.mapto_refine2[:,1]))%2
        
        self.kernel_size = kernel_size
        self.stride = stride
        


    def forward(self, input):
        
        batchsize, filters = input.shape[:2]
        
        # if we want to change average to something else, change the following lines
        # work with a flat tensor (tmp). write new values there
        # center vertices
        tmp = input[:,:,self.basex,self.basey]
        # sum vertex values in same column
        tmp += input[:,:,self.basexminus1,self.basey]
        tmp += input[:,:,self.basexplus1,self.basey]
        # same row
        tmp += input[:,:,self.basex,self.baseyplus1]
        tmp += input[:,:,self.basex,self.baseyminus1]
        # lower row, because of the uneven kernel size
        tmp += input[:,:,self.basexplus1,self.baseyplus1]
        tmp += input[:,:,self.basexplus1,self.baseyminus1]
        # average
        tmp  = 1./7 * tmp

        # write the pooling results to the output tensor
        height_padded=3
        width_padded=3
        data_out = torch.zeros((batchsize, filters, height_padded, width_padded), device=input.device)
        data_out[:,:,self.mapto_refine2[:,0],self.mapto_refine2[:,1]] = tmp

        return data_out

#first depooling
class first_depooling(nn.Module):
    r"""
    input: 3x3 (kernel size 0)
    output: 7x7 (kernel size 1)
    """

    def __init__(
        self, kernel_size=1, stride=1
    ):
        super(first_depooling, self).__init__()
        
        # to these vertices we can copy a value: base_vertices_at_refine_2
        # positions of the known vertices in mesh that is not yet refined: mapto_refine2

        # size of output patches
        self.height_padded=7
        self.width_padded=7

        # size of input patches
        self.height_padded_in=3
        self.width_padded_in=3
        
        self.base_vertices_at_refine2 = torch.tensor(np.array([[1,1],[3,1],[5,1],
                                                        [2,3],[4,3],
                                                        [3,5]]))
        
        self.mapto_refine2 = torch.zeros(self.base_vertices_at_refine2.shape).long() 
        self.mapto_refine2[:,0] = torch.floor(self.base_vertices_at_refine2[:,0]/2)  
        self.mapto_refine2[:,1] = torch.floor((self.base_vertices_at_refine2[:,1])/2)
        self.mapto_refine2[:,0] -= ((self.mapto_refine2[:,1]))%2
        
        self.depool1_verts_evencc =  torch.tensor(np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0],  [6, 0],
                         [1, 2], [2, 2], [3, 2], [4, 2], [5, 2],  [6, 2],
                         [2, 4], [3, 4], [4, 4], [5, 4],
                         [3, 6], [4, 6]])).long()
        
        # take the left, upper left, right, upper right vertex of sparse tensor and take the average of all nonzero elements
        self.depool1_verts_evencc_avg = torch.tensor([[[ii-1,max(jj-1,0)],
                                            [ii-1,min(jj+1,self.width_padded-1)],
                                            [ii,max(jj-1,0)],
                                            [ii,min(jj+1,self.width_padded-1)]] for ii,jj in self.depool1_verts_evencc]).long()
        
        self.depool1_verts_unevencc = torch.tensor(np.array([[0, 1], [2, 1], [4, 1], [6, 1], 
                          [1, 3], [3, 3], [5, 3],
                          [2, 5], [4, 5]])).long()
        
        self.depool1_verts_unevencc_half = (self.depool1_verts_unevencc/2).long()
        
        self.kernel_size = kernel_size
        self.stride = stride
        


    def forward(self, input):
        
        # size of output patches
        data_out = torch.zeros((input.shape[0],input.shape[1],self.height_padded,self.width_padded), device=input.device)
        # copy known vertices
        data_out[:,:,self.base_vertices_at_refine2[:,0],
                 self.base_vertices_at_refine2[:,1]] = input[:,:,self.mapto_refine2[:,0],self.mapto_refine2[:,1]]

                
        #######################
        # even columns: take average using the tensor data_out2
        index = self.depool1_verts_evencc.long()

        tmp = (data_out[:,:,self.depool1_verts_evencc_avg[:,0,0],self.depool1_verts_evencc_avg[:,0,1]]).unsqueeze(0)
        # tmp.shape: 1,#batch,#filter,#vertices
        for ii in range(1,4):
            # stack the values for the other neighboring vertices
            tmp = torch.cat((tmp,data_out[:,:,self.depool1_verts_evencc_avg[:,ii,0],self.depool1_verts_evencc_avg[:,ii,1]].unsqueeze(0)))
        # if we want to change average to something else, change the following lines    
        # tmp.shape: 4,#batch,#filter,#vertices
        # sum the values although some might be zero
        tmp2 = tmp.sum(axis=0)
        # take the average of all nonzero values (axis = 0, the stacked axis) 
        #   count the quantity of nonzero values, if zero use one
        tmp = torch.sum(tmp[:,0,0]!=0,axis=0)
        data_out[:,:,self.depool1_verts_evencc[:,0],self.depool1_verts_evencc[:,1]] = tmp2/(torch.max(torch.stack((tmp,tmp*0+1)),dim=0)[0])

        #####################
        # uneven columns
        
        # stack values that have to be average for each new vertex in uneven columns
        tmp = torch.stack((input[:,:,torch.minimum(self.depool1_verts_unevencc_half[:,0],
                                         (self.height_padded_in-1)*torch.ones(len(self.depool1_verts_unevencc_half))).long(),
                                    self.depool1_verts_unevencc_half[:,1]], 
                           input[:,:,torch.maximum(self.depool1_verts_unevencc_half[:,0]-1, 
                                         0*torch.ones(len(self.depool1_verts_unevencc_half))).long(),
                                    self.depool1_verts_unevencc_half[:,1]]))

        # if we want to change average to something else, change the following lines 

        # tmp.shape: 2,#batch,#filter,#vertices
        # sum the values although some might be zero
        tmp2 = tmp.sum(axis=0)
        # take the average of all nonzero values (axis = 0, the stacked axis) 
        #   count the quantity of nonzero values, if zero use one
        tmp = torch.sum(tmp[:,0,0]!=0,axis=0)
        data_out[:,:,self.depool1_verts_unevencc[:,0],self.depool1_verts_unevencc[:,1]] = tmp2/(torch.max(torch.stack((tmp,tmp*0+1)),dim=0)[0])

        return data_out


#first depooling
class second_depooling(nn.Module):
    r"""
    input: 7x7 (kernel size 1)
    output: 13x13 (kernel size 2)
    """

    def __init__(
        self, kernel_size=2, stride=1
    ):
        super(second_depooling, self).__init__()
        
        # to these vertices we can copy a value: base_vertices_at_refine_2
        # positions of the known vertices in mesh that is not yet refined: mapto_refine2

        # size of output patches
        self.height_padded=13
        self.width_padded=13

        # size of input patches
        self.height_padded_in=7
        self.width_padded_in=7
        
        self.base_vertices_at_refine3 = torch.tensor(np.array([[1,0],[3,0],[5,0],[7,0],[9,0],[11,0],
                            [0,2],[2,2],[4,2],[6,2],[8,2],[10,2],[12,2],
                            [1,4],[3,4],[5,4],[7,4],[9,4],[11,4],
                            [2,6],[4,6],[6,6],[8,6],[10,6],
                            [3,8],[5,8],[7,8],[9,8],
                            [4,10],[6,10],[8,10],
                            [5,12],[7,12]]))
        
        self.mapto_refine3 = torch.zeros(self.base_vertices_at_refine3.shape).long() 
        self.mapto_refine3[:,0] = torch.floor(self.base_vertices_at_refine3[:,0]/2)  
        self.mapto_refine3[:,1] = torch.floor((self.base_vertices_at_refine3[:,1])/2)
        self.mapto_refine3[:,0] += ((self.mapto_refine3[:,1])+1)%2
        
        self.depool2_verts_evencc =  torch.tensor(np.array([[ 4,  0], [ 6,  0], [10,  0], [ 2,  0], [ 8,  0],
                         [ 5,  2], [ 7,  2], [ 3,  2], [ 9,  2], [ 1,  2], [11,  2],
                         [ 2,  4], [ 8,  4], [10,  4], [ 6,  4], [ 4,  4],
                         [ 7,  6], [ 9,  6], [ 5,  6], [ 3,  6],
                         [ 4,  8], [ 6,  8], [ 8,  8], 
                         [ 5, 10], [ 7, 10],
                         [ 6, 12]])).long()
        
        self.depool2_verts_evencc_half = (self.depool2_verts_evencc/2).long()
        

        
        self.depool2_verts_unevencc = torch.tensor(np.array([[ 5,  1], [ 6,  1], [ 7,  1], [ 3,  1], [ 0,  1], [ 4,  1], [ 9,  1], [ 2,  1], [10,  1], [ 1,  1], [11,  1], [ 8,  1],
                          [ 6,  3], [ 3,  3], [ 7,  3], [ 4,  3], [ 8,  3], [ 2,  3], [ 9,  3], [ 1,  3], [10,  3], [ 0,  3], [11,  3], [ 5,  3],
                          [ 6,  5], [ 4,  5], [10,  5], [ 1,  5], [ 9,  5], [ 5,  5], [ 2,  5], [ 8,  5], [ 7,  5], [ 3,  5],
                          [ 4,  7], [ 6,  7], [ 9,  7], [ 5,  7], [ 8,  7], [ 3,  7], [ 7,  7], [ 2,  7],
                          [ 6,  9], [ 5,  9], [ 7,  9], [ 8,  9], [ 3,  9], [ 4,  9],
                          [ 4, 11], [ 7, 11], [ 5, 11], [ 6, 11]])).long()
        
        # take the left, lower left, right, lower right vertex of sparse tensor and take the average of all nonzero elements
        self.depool2_verts_unevencc_avg = torch.tensor([[[ii,max(jj-1,0)],
                                               [ii,min(jj+1,self.width_padded-1)],
                                               [min(ii+1,self.height_padded-1),max(jj-1,0)],
                                               [min(ii+1,self.height_padded-1),min(jj+1,self.width_padded-1)]] for ii,jj in self.depool2_verts_unevencc]).long()
        
        
        self.kernel_size = kernel_size
        self.stride = stride
        


    def forward(self, input):
        
        # size of output patches
        data_out = torch.zeros((input.shape[0],input.shape[1],self.height_padded,self.width_padded), device=input.device)
        # copy known vertices
        data_out[:,:,self.base_vertices_at_refine3[:,0],
                 self.base_vertices_at_refine3[:,1]] = input[:,:,self.mapto_refine3[:,0],self.mapto_refine3[:,1]]

        #####################
        # uneven columns: take average using the tensor data_out

        tmp = (data_out[:,:,self.depool2_verts_unevencc_avg[:,0,0],self.depool2_verts_unevencc_avg[:,0,1]]).unsqueeze(0)
        # tmp.shape: 1,#batch,#filter,#vertices
        for ii in range(1,4):
            # stack the values for the other neighboring vertices
            tmp = torch.cat((tmp,data_out[:,:,self.depool2_verts_unevencc_avg[:,ii,0],self.depool2_verts_unevencc_avg[:,ii,1]].unsqueeze(0)))
        # if we want to change average to something else, change the following lines    
        # tmp.shape: 4,#batch,#filter,#vertices
        # sum the values although some might be zero
        tmp2 = tmp.sum(axis=0)
        # take the average of all nonzero values (axis = 0, the stacked axis) 
        #   count the quantity of nonzero values, if zero use one
        tmp = torch.sum(tmp[:,0,0]!=0,axis=0)
        data_out[:,:,self.depool2_verts_unevencc[:,0],self.depool2_verts_unevencc[:,1]] = tmp2/(torch.max(torch.stack((tmp,tmp*0+1)),dim=0)[0])       
        

        #####################
        # even columns
        
        # stack values that have to be average for each new vertex in uneven columns
        tmp = torch.stack((input[:,:,torch.minimum(self.depool2_verts_evencc_half[:,0],
                                         (self.height_padded_in-1)*torch.ones(len(self.depool2_verts_evencc_half))).long(),
                                    self.depool2_verts_evencc_half[:,1]], 
                           input[:,:,torch.minimum(self.depool2_verts_evencc_half[:,0]+1, 
                                         (self.height_padded_in-1)*torch.ones(len(self.depool2_verts_evencc_half))).long(),
                                    self.depool2_verts_evencc_half[:,1]]))

        # if we want to change average to something else, change the following lines 

        # tmp.shape: 2,#batch,#filter,#vertices
        # sum the values although some might be zero
        tmp2 = tmp.sum(axis=0)
        # take the average of all nonzero values (axis = 0, the stacked axis) 
        #   count the quantity of nonzero values, if zero use one
        tmp = torch.sum(tmp[:,0,0]!=0,axis=0)
        data_out[:,:,self.depool2_verts_evencc[:,0],self.depool2_verts_evencc[:,1]] = tmp2/(torch.max(torch.stack((tmp,tmp*0+1)),dim=0)[0])         


        return data_out

class Hexconv_Autoencoder(nn.Module):

    def __init__(self, hid_rep=10):
        super(Hexconv_Autoencoder, self).__init__()
        in_channels1, out_channels1 = 3, 16
        self.out_channels2 =  32
        self.hid_rep = hid_rep
        kernel_size1, kernel_size2, stride = 2, 1, 1
        #stride_maxpool1 = 3
        #stride_maxpool2 = 2
        
        # 3 input image channel, 6 output channels, 2x2 hexconv convolution, stride for conv is always 1. reduce dimension via maxpool
        self.hexconv1 = Conv2d(in_channels1, out_channels1, kernel_size1, stride, bias=False)
        self.avgpool1 = first_pooling() # output 7x7
        
        self.hexconv2 = Conv2d(out_channels1, self.out_channels2, kernel_size2, stride, bias=False)
        self.avgpool2 = second_pooling() # output 3x3

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.out_channels2 * 3 * 3, self.hid_rep) 
        #self.fc2 = nn.Linear(84, self.hid_rep)
        
        # DECONV
        self.defc1 = nn.Linear(self.hid_rep, self.out_channels2 * 3 * 3) #, 84)
        #self.defc2 = nn.Linear(84, self.out_channels2 * 4 * 10)
        
        self.upsample1 = first_depooling()
        self.hexdeconv1 = Conv2d(self.out_channels2, out_channels1, kernel_size2, stride, bias=False)
        
        self.upsample2 = second_depooling()
        self.hexdeconv2 = Conv2d(out_channels1, out_channels1, kernel_size1, stride, bias=False)
        
        self.hexdeconv3 = Conv2d(out_channels1, in_channels1, 1, stride, bias=False)
          
        
        #plot_hextensor(upsampled_output3, figname='tensor',figsize=(20,20))
        
        #self.defc2 = nn.Linear(in_channels1 * 24 * 60, in_channels1 * 24 * 60)
        
        

    def forward(self, x):
        
        x = self.avgpool1(F.relu(self.hexconv1(x)))
        x = self.avgpool2(F.relu(self.hexconv2(x)))
        
        # reshape
        x = x.view(-1, self.out_channels2 * 3 * 3)
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
                
        x = self.fc1(x)
                
        # DECONV
        
        y = F.relu(self.defc1(x))
        #y = F.relu(self.defc2(y))
        
        # reshape
        y = y.view(-1, self.out_channels2, 3, 3)
                
        y = F.relu(self.hexdeconv1(self.upsample1(y)))
        y = F.relu(self.hexdeconv2(self.upsample2(y)))
        y = self.hexdeconv3(y) #F.relu(self.hexdeconv3(y))
        
        return x, y
