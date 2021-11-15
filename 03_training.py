import numpy as np
import torch
import torch.nn.functional as F

import os
import os.path as osp
import pickle
import argparse
import time

from networks.Hexconv_Autoencoder import Hexconv_Autoencoder

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import utils

###########################
### variable definition ###

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment.')
parser.add_argument('--model_name', type=str, default='test', help='Name of the model and training.')
parser.add_argument('--dataset', type=str, default='gallop', help='Name of the dataset.')
parser.add_argument('--device_idx', type=str, default='0', help='Device. Can be CPU or the id of the GPU.')

# mesh refinement
parser.add_argument('--refine', type=int, default=3, help='Level of refinement. Tested for 3.')

# patch arguments
parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size of the first convolutional layer. Tested for 2.') # this is also the size of the padding
parser.add_argument('--patch_zeromean', type=bool, default=True, help='Move the patches to zero mean: True/False.')

# training variables
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--hid_rep', type=int, default=8, help='Size of the hidden representation.')
parser.add_argument('--lr', type=float, default=0.001, help='Leraning Rate.')
parser.add_argument('--Niter', type=int, default=501, help='Number of Epochs.')

# others
parser.add_argument('--seed', type=int, default=1, help='Seed')

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.data_train_patches = osp.join(args.data_fp, 'train_patches_{}'.format(args.exp_name))
args.data_training_results = osp.join(args.data_fp, 'train_patches_{}'.format(args.exp_name), 'output')
args.model_fp = osp.join(args.work_dir, 'model', args.dataset)
utils.mkdirs(args.model_fp)
utils.mkdirs(args.data_training_results)

# set device to either cpu (local machine) or gpu (cluster)
if args.device_idx != 'cpu':
    device = torch.device('cuda', int(args.device_idx))
    print('device cuda', args.device_idx)
else:
    device = torch.device('cpu')
    print('device', args.device_idx)
#torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

model_logs = osp.join(args.model_fp, 'logs')
utils.mkdirs(model_logs)

# write arguments to a log file
with open(model_logs+'/arguments_03_training_exp_{}_model_{}.txt'.format(args.exp_name, args.model_name), 'w') as f:
    for aa in list(args.__dict__.keys()):
        if args.work_dir in str(args.__dict__[aa]):
            text = '(path) {}: {}\n'.format(aa, args.__dict__[aa])
        else:
            text = '--{} {}\n'.format(aa, args.__dict__[aa])
        f.write(text)
        print(text, end='')



patch_indices = np.array([[2, 2], [2, 3], 
                [3, 2], [3, 3], [3, 4], [3, 5], 
                [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7],  
                [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],  
                [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], 
                [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], 
                [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], 
                [9, 2], [9, 3], [9, 4],  
                [10, 2]])
mask_size=len(patch_indices)
loss_mask_refine3 = torch.zeros((1,1,13,13),device=device)
loss_mask_refine3[:,:,patch_indices[:,0],patch_indices[:,1]]=1

####
# where to save the model

model_path = args.model_fp + '/model_{}_{}.pt'.format(args.exp_name, args.model_name)


#### load training patches
    
with open(args.data_train_patches+'/all_data.npy', 'rb') as f:
    all_data  = torch.tensor(np.load(f)).float()
with open(args.data_train_patches+'/train_index.npy', 'rb') as f:
    train_index = np.load(f)
with open(args.data_train_patches+'/test_index.npy', 'rb') as f:
    test_index  = np.load(f)



if args.patch_zeromean == True:
    print("Move all patches to zero mean")
    meani = (torch.mean(all_data,axis=(2,3)))
    meani = meani.repeat_interleave(all_data.shape[2]*all_data.shape[3],dim=1)
    meani = torch.reshape(meani,all_data.shape)
    all_data = all_data - meani
    
test_data = all_data[test_index]
train_data = all_data[train_index]

train_data_cuda = train_data.to(device)


#### create train and test loader
training_set = torch.utils.data.TensorDataset(train_data) 
trainloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, pin_memory=True)
testing_set = torch.utils.data.TensorDataset(train_data) 
testloader = torch.utils.data.DataLoader(testing_set , batch_size=112,
                                          shuffle=False) 

print(train_data.shape[0], 'training samples')
print(test_data.shape[0], 'testing samples')

if args.device_idx != 'cpu':
    network = Hexconv_Autoencoder(hid_rep=args.hid_rep).cuda(device=device)
else:
    network = Hexconv_Autoencoder(hid_rep=args.hid_rep)
    

params = list(network.parameters())
#print(len(params))
trainable_weights = 0
for ii in range(len(params)):
    #print(params[ii].size())
    trainable_weights += (np.prod(params[ii].size()))
print(trainable_weights,'trainable weights')
print(np.prod(train_data.size()),'dimensional input')

import torch.optim as optim

# loss function
criterion = nn.MSELoss(reduction='sum')
# create your optimizer
optimizer = optim.Adam(network.parameters(), lr=args.lr)

all_loss = []

mylogfile = open(model_logs+'/arguments_03_training_exp_{}_model_{}.txt'.format(args.exp_name, args.model_name), "a")
mylogfile.write("\n-----------------\nLOG \n-----------------\n\n")

mylogfile.write('{} training samples\n'.format(train_data.shape[0]))
mylogfile.write('{} testing samples\n'.format(test_data.shape[0]))
mylogfile.write('{} trainable weights\n'.format(trainable_weights))
mylogfile.write('{} dimensional input\n\n'.format(np.prod(train_data.size())))

start_time = time.time()
start_time_epoch = time.time()
for epoch in range(args.Niter):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs]
        inputs = data[0].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize          
        _, outputs = network(inputs)
        
        loss = criterion(loss_mask_refine3*outputs, loss_mask_refine3*inputs)/(args.batch_size*mask_size)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    if epoch%10 == 0:
        print('Epoch {:3}'.format(epoch), ', Loss: {:10f}'.format(loss.item() ), ', Time: {:8f} seconds'.format(time.time() - start_time_epoch))
        mylogfile.write('Epoch {:3}'.format(epoch))
        mylogfile.write(', Loss: {:10f}'.format(loss.item() ))
        mylogfile.write(', Time: {:8f} seconds \n'.format(time.time() - start_time_epoch))
        start_time_epoch = time.time()
        
    all_loss.append(loss.item())

print('Finished Training in {} seconds'.format(np.round(time.time()-start_time,4)))
mylogfile.write('Finished Training in {} seconds \n\n'.format(np.round(time.time()-start_time,4)))


### plot the loss
plt.plot(np.arange(len(all_loss)), all_loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylim(0,0.005)
plt.savefig(model_path[:-3]+'_training_loss.png')
plt.show()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in network.state_dict():
    print(param_tensor, "\t", network.state_dict()[param_tensor].size())

torch.save(network.state_dict(), model_path)

### First look at results
import torch.nn.functional as F

net_cpu = network.cpu()

emb_train, output_train = net_cpu(train_data)
emb_test, output_test = net_cpu(test_data)

loss_mask_refine3_cpu = loss_mask_refine3.detach().cpu()

print('Errors minimized by optimizer')
print('Training error: {:8f}'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_train, loss_mask_refine3_cpu*train_data).item()*(13*13)/mask_size*3,10)))
print('Testing error:  {:8f}'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_test, loss_mask_refine3_cpu*test_data).item()*(13*13)/mask_size*3,10)))

mylogfile.write('Training error: {:8f} \n'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_train, loss_mask_refine3_cpu*train_data).item()*(13*13)/mask_size*3,10)))
mylogfile.write('Testing error:  {:8f} \n'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_test, loss_mask_refine3_cpu*test_data).item()*(13*13)/mask_size*3,10)))


print()
print('Minimum training input :',torch.min((train_data)).item())
print('Maximum training input :',torch.max((train_data)).item())
print('Minimum training output:',torch.min((loss_mask_refine3_cpu*output_train)).item())
print('Maximum training output:',torch.max((loss_mask_refine3_cpu*output_train)).item())

print('\nMinimum testing input :',torch.min((test_data)).item())
print('Maximum testing input :',torch.max((test_data)).item())
print('Minimum testing output:',torch.min((loss_mask_refine3_cpu*output_test)).item())
print('Maximum testing output:',torch.max((loss_mask_refine3_cpu*output_test)).item())


### save the results
all_emb = torch.empty((emb_train.shape[0] + emb_test.shape[0], emb_train.shape[1]))
all_emb[test_index] = emb_test
all_emb[train_index] = emb_train

all_output = torch.empty((output_train.shape[0] + output_test.shape[0], *(output_train.shape[1:])))
all_output[test_index] = output_test
all_output[train_index] = output_train

if args.patch_zeromean == True:
    print("Translate all patches back to original position")
    all_output = all_output + meani

save_path_output_data = args.data_training_results+'/'
with open(save_path_output_data+'all_emb_{}.npy'.format(args.model_name) , 'wb') as f:
    np.save(f, all_emb.detach().numpy())
    
with open(save_path_output_data+'all_output_{}.npy'.format(args.model_name) , 'wb') as f:
    np.save(f, all_output.detach().numpy())

mylogfile.close()