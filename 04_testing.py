import numpy as np
import torch
import torch.nn.functional as F

import os
import os.path as osp
import pickle
import argparse

import igl

from mesh import clean_mesh_get_edges
from networks.Hexconv_Autoencoder import Hexconv_Autoencoder

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import utils, normalize

from sklearn.metrics import mean_squared_error

###########################
### variable definition ###

parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment.')
parser.add_argument('--model_name', type=str, default='test', help='Name of the model and training.')
parser.add_argument('--dataset', type=str, default='gallop', help='Name of the dataset.')
parser.add_argument('--device_idx', type=str, default='0', help='Device. Can be CPU or the id of the GPU.')

# data split
parser.add_argument('--test_split', nargs="+", type=str, default=['elephant'], help='List of test samples. For those the train-test-split is 0-100.') #type=str, default='elephant') # train-test-split: 0-100 for test samples
parser.add_argument('--test_version', nargs="+", type=str, default=[], help='Only for car crash. versions that are completly part of test data.') # only for car crash
parser.add_argument('--test_ratio', type=float, default=0.25, help='train-test-split. Default: 75% training, 25% testing plus the test samples given by variable test_split.') # train-test-split: 75-25 for train samples

# mesh refinement
parser.add_argument('--refine', type=int, default=3, help='Level of refinement. Tested for 3.')

# patch arguments
parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size of the first convolutional layer. Tested for 2.') # this is also the size of the padding
parser.add_argument('--patch_zeromean', type=bool, default=True, help='Move the patches to zero mean: True/False.')

# training variables
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--hid_rep', type=int, default=8, help='Size of the hidden representation.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')
parser.add_argument('--Niter', type=int, default=501, help='Number of Epochs.')

# others
parser.add_argument('--seed', type=int, default=1, help='Seed')

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.data_raw = osp.join(args.data_fp, 'raw')
args.data_preprocessed = osp.join(args.data_fp, 'preprocessed')
args.data_semireg = osp.join(args.data_fp, 'semiregular')
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
#torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True


model_logs = osp.join(args.model_fp, 'logs')


mylogfilename = model_logs+'/arguments_03_training_exp_{}_model_{}.txt'.format(args.exp_name, args.model_name)

if osp.isfile(mylogfilename):
    mylogfile = open(mylogfilename, "a")
else:
    utils.mkdirs(model_logs)
    mylogfile = open(mylogfilename, "w")
mylogfile.write("\n-----------------\nLOG \n-----------------\n\n")
    
test_versions = []  ## add posibly versions to this list if wanted
if len(args.test_version) > 0:
    test_versions = args.test_version #['sim_041','sim_049']
    print('Test Versions:', test_versions)
    mylogfile.write("Test Versions: {}\n".format(test_versions))


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

#train_data = train_data[:1000:100]
#test_data = test_data[:1000:100]

#train_data_cuda = train_data.to(device)


#### create train and test loader
#training_set = torch.utils.data.TensorDataset(train_data) #, train_label)
#trainloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, 
#                                          num_workers=2) # sampler=sampler, pin_memory=False)
#testing_set = torch.utils.data.TensorDataset(train_data) #test_data) #, test_label)
#testloader = torch.utils.data.DataLoader(testing_set , batch_size=112,
#                                          shuffle=False) #, pin_memory=False)

print(train_data.shape[0], 'training samples')
print(test_data.shape[0], 'testing samples')

#if args.device_idx != 'cpu':
#    network = Hexconv_Autoencoder(hid_rep=args.hid_rep).cuda()
#else:
    
network = Hexconv_Autoencoder(hid_rep=args.hid_rep)
    

network.load_state_dict(torch.load(model_path))
network.eval()

def count_parameters(model):
    print("Modules", "Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print(name, param)
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(network)


save_path_output_data = args.data_training_results+'/'

if not os.path.isfile(save_path_output_data+'all_output_{}.npy'.format(args.model_name)):

    ### First look at results
    import torch.nn.functional as F

    net_cpu = network #.cpu()

    #emb_train = np.zeros((train_data.shape[0],args.hid_rep))
    #output_train = np.zeros(train_data.shape)

    #steps = int(len(train_data)/5)
    #for ii in range(4):
    #    emb_train[ii*steps:(ii+1)*steps], output_train[ii*steps:(ii+1)*steps] = net_cpu(train_data[ii*steps:(ii+1)*steps])
    #ii=5
    #emb_train[ii*steps:], output_train[ii*steps:] = net_cpu(train_data[ii*steps:])

    print('Errors minimized by optimizer')
    loss_mask_refine3_cpu = loss_mask_refine3.detach().cpu()


    if train_data.shape[0] != 0:
        emb_train, output_train = net_cpu(train_data)
        print('Training error: {:8f}'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_train, loss_mask_refine3_cpu*train_data).item()*(13*13)/mask_size*3,10)))

    emb_test, output_test = net_cpu(test_data)
    print('Testing error:  {:8f}'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_test, loss_mask_refine3_cpu*test_data).item()*(13*13)/mask_size*3,10)))

    print()
    if train_data.shape[0] != 0:
        print('Minimum training input :',torch.min((train_data)).item())
        print('Maximum training input :',torch.max((train_data)).item())
        print('Minimum training output:',torch.min((loss_mask_refine3_cpu*output_train)).item())
        print('Maximum training output:',torch.max((loss_mask_refine3_cpu*output_train)).item())

    print('\nMinimum testing input :',torch.min((test_data)).item())
    print('Maximum testing input :',torch.max((test_data)).item())
    print('Minimum testing output:',torch.min((loss_mask_refine3_cpu*output_test)).item())
    print('Maximum testing output:',torch.max((loss_mask_refine3_cpu*output_test)).item())

    ### save the results

    all_emb = torch.empty((train_data.shape[0] + emb_test.shape[0], emb_test.shape[1]))
    all_emb[test_index] = emb_test


    all_output = torch.empty((train_data.shape[0] + output_test.shape[0], 
                              train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    all_output[test_index] = output_test
    if train_data.shape[0] != 0:
        all_emb[train_index] = emb_train
        all_output[train_index] = output_train
        
    if args.patch_zeromean == True:
        print("Translate all patches back to original position")
        all_output = all_output + meani

    with open(save_path_output_data+'all_emb_{}.npy'.format(args.model_name) , 'wb') as f:
        np.save(f, all_emb.detach().numpy())

    with open(save_path_output_data+'all_output_{}.npy'.format(args.model_name) , 'wb') as f:
        np.save(f, all_output.detach().numpy())



save_path_output_data = args.data_training_results+'/'
with open(save_path_output_data+'all_emb_{}.npy'.format(args.model_name), 'rb') as f:
    all_emb = np.load(f)

emb_test = torch.tensor(all_emb[test_index])
emb_train = torch.tensor(all_emb[train_index])

print(emb_train.shape[0], 'training embedding')
print(emb_test.shape[0], 'testing embedding')

with open(save_path_output_data+'all_output_{}.npy'.format(args.model_name), 'rb') as f:
    all_output = np.load(f)
    
if args.patch_zeromean == True:
    output_test = torch.tensor((all_output - np.asarray(meani))[test_index])
    output_train = torch.tensor((all_output - np.asarray(meani))[train_index])
else:
    output_test = torch.tensor(all_output[test_index])
    output_train = torch.tensor(all_output[train_index])

import torch.nn.functional as F


loss_mask_refine3_cpu = loss_mask_refine3.detach().cpu()


print()
if train_data.shape[0] != 0:
    print('Minimum training input :',torch.min((train_data)).item())
    print('Maximum training input :',torch.max((train_data)).item())
    mylogfile.write('\nMinimum training input : {}\n'.format(torch.min((train_data)).item()))
    mylogfile.write('Maximum training input : {}\n'.format(torch.max((train_data)).item()))
    print('Minimum training output:',torch.min((loss_mask_refine3_cpu*output_train)).item())
    print('Maximum training output:',torch.max((loss_mask_refine3_cpu*output_train)).item())
    mylogfile.write('Minimum training output : {}\n'.format(torch.min((loss_mask_refine3_cpu*output_train)).item()))
    mylogfile.write('Maximum training output : {}\n'.format(torch.max((loss_mask_refine3_cpu*output_train)).item()))

print('\nMinimum testing input :',torch.min((test_data)).item())
print('Maximum testing input :',torch.max((test_data)).item())
print('Minimum testing output:',torch.min((loss_mask_refine3_cpu*output_test)).item())
print('Maximum testing output:',torch.max((loss_mask_refine3_cpu*output_test)).item())
mylogfile.write('Minimum testing input : {}\n'.format(torch.min((test_data)).item()))
mylogfile.write('Maximum testing input : {}\n'.format(torch.max((test_data)).item()))
mylogfile.write('Minimum testing output : {}\n'.format(torch.min((loss_mask_refine3_cpu*output_test)).item()))
mylogfile.write('Maximum testing output : {}\n\n'.format(torch.max((loss_mask_refine3_cpu*output_test)).item()))


print('Errors minimized by optimizer')
if train_data.shape[0] != 0:
    print('Training error: {:8f}'.format((F.mse_loss(loss_mask_refine3_cpu*output_train, loss_mask_refine3_cpu*train_data).item()*(13*13)/mask_size*3)))
print('Testing error:  {:8f}'.format((F.mse_loss(loss_mask_refine3_cpu*output_test, loss_mask_refine3_cpu*test_data).item()*(13*13)/mask_size*3)))

if train_data.shape[0] != 0:
    text = '\nTraining error: {:8f}\n'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_train, loss_mask_refine3_cpu*train_data).item()*(13*13)/mask_size*3,10))
    mylogfile.write(text)
text = 'Testing error:  {:8f}\n'.format(round(F.mse_loss(loss_mask_refine3_cpu*output_test, loss_mask_refine3_cpu*test_data).item()*(13*13)/mask_size*3,10))
mylogfile.write(text)


## versions
versions = [f.name for f in os.scandir(args.data_raw) if f.is_dir() and 'checkpoints' not in f.name]
print('Versions:', versions)

## parts/samples (for every version the same!)
samples = [f.name for f in os.scandir(osp.join(args.data_raw, versions[0])) if f.is_dir() and 'checkpoints' not in f.name]
print('Samples:', samples)
test_samples = [sa for sa in samples if sa in args.test_split]
#if 'faust8' in test_samples:
#    test_samples += ['faust9']
print('\nTest Sample:', test_samples)
train_samples = [sa for sa in samples if sa  not in test_samples] 
print('Train Samples:', train_samples, '\n')

data_index = 0

mse2_total=[]
mse2_total_test=[]

####
# set the mesh back together (from patches to total mesh)

for pp, pname in enumerate(train_samples+test_samples):
    print('################')
    print('Sample', pname)
    mylogfile.write('\n################\nSample {}\n'.format(pname))
    
    data_preprocessed_part = osp.join(args.data_preprocessed, pname)
    if 'norot' in args.exp_name or 'inter' in args.exp_name:
        base_part = os.path.join(data_preprocessed_part, "mesh_{}_remesh_exp_{}.obj".format(pname, args.exp_name[:-6]))
    else:
        base_part = os.path.join(data_preprocessed_part, "mesh_{}_remesh_exp_{}.obj".format(pname, args.exp_name))
    data_semireg_part = osp.join(args.data_semireg, pname)
    
    # this is necessary for the projection
    if 'norot' in args.exp_name or 'inter' in args.exp_name:
        semiregular_mesh = os.path.join(data_semireg_part, 
                  '{}_remesh_exp_{}_refinelevel_{}.obj'.format(pname,args.exp_name[:-6],args.refine))
    else:
        semiregular_mesh = os.path.join(data_semireg_part, 
                  '{}_remesh_exp_{}_refinelevel_{}.obj'.format(pname,args.exp_name,args.refine))
    VV, _, _, FF, _, _ = igl.read_obj(semiregular_mesh)   
    print('  -> remeshed mesh vertices:',VV.shape)
    
    ##### Load Base Data    
    VV_base, _, _, FF_base, _, _ = igl.read_obj(base_part) 
    _, FF_base, EE_base, boundary_edges, boundary_vertices, nonmanifold_edges = clean_mesh_get_edges(VV_base, FF_base)
    
    N_triangles = len(FF_base)
    
    # use the results from rotation = 0
    rotation = 0
    
    with open(args.data_train_patches+'/padded_hexagonal_vertex_ids_sample_{}.npy'.format(pname), 'rb') as f:
        padded_hexagonal_vertex_ids = np.load(f)[:,0,0]
    print('  -> Patch to Vertex ID:', padded_hexagonal_vertex_ids.shape)
    

    count_part_patches = 0
    
    ### print the errors per part
    mse2=[]
    mse2_test=[]
    
    for kk,version in enumerate(versions):

            if 'norot' in args.exp_name or 'inter' in args.exp_name:
                pickle_data = os.path.join(data_semireg_part, 
                                "projected_mesh_{}_{}_remesh_exp_{}_refinelevel_{}_vertex_values.p".format(pname, 
                                        version, args.exp_name[:-6], args.refine))
            else:
                pickle_data = os.path.join(data_semireg_part, 
                                "projected_mesh_{}_{}_remesh_exp_{}_refinelevel_{}_vertex_values.p".format(pname, 
                                        version, args.exp_name, args.refine))

            with open(pickle_data, "rb") as file:
                projected_VV = pickle.load(file)
                
            if 'car' not in args.dataset:
                meshfiles = [f.name for f in os.scandir(osp.join(args.data_raw, version, pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
                meshfiles.sort()

                select_tt = np.arange(0,len(meshfiles),1)

                if args.test_ratio > 0:
                    test_tt = select_tt[-int(len(meshfiles)* args.test_ratio ):] # use the last int(len(meshfiles)* args.test_ratio ) timesteps for testing
                else:
                    test_tt = []
            
            else:
                # raw data is not uploaded yet. extract the number of timesteps from the projected vertices. projected_VV
                meshfiles = list(np.arange(projected_VV.shape[0]))
                select_tt = np.arange(0,projected_VV.shape[0],1)

                if args.test_ratio > 0:
                    test_tt = select_tt[-int(projected_VV.shape[0]* args.test_ratio ):] # use the last int(len(meshfiles)* args.test_ratio ) timesteps for testing
                elif args.test_ratio < 0:
                    ## select specific timesteps, save them for reproducibility reasons 
                    test_number_tt = int(projected_VV.shape[0] * -args.test_ratio)
                    
                    file_random_ts = args.data_preprocessed+"/random_test_timesteps.txt"
                    
                    if osp.isfile(file_random_ts):
                        with open(file_random_ts, 'r') as f:
                            test_tt = [int(line.rstrip('\n')) for line in f]
                    
                    else:
                        test_tt = np.random.choice(projected_VV.shape[0], test_number_tt, replace=False)
                        with open(file_random_ts, 'w') as f:
                            for s in test_tt:
                                f.write(str(s) + '\n')

                    if kk == 0 and rotation == 0:
                        print('    Random test timestps:', test_tt)
                else:
                    test_tt = []
            

            output_part_version = np.swapaxes(np.swapaxes(np.reshape(all_output[data_index  + rotation * len(meshfiles)*N_triangles : data_index + (rotation+1) * len(meshfiles)*N_triangles],
                                                 (len(meshfiles),-1,3,13,13)), 2, 3), 3, 4)
            
            # get the corresponding outputs and undo the patchwise normalization
            if args.patch_zeromean == True:
            
                input_part_version = np.swapaxes(np.swapaxes(np.reshape((all_data+meani)[data_index  + rotation * len(meshfiles)*N_triangles : data_index + (rotation+1) * len(meshfiles)*N_triangles],
                                     (len(meshfiles),-1,3,13,13)), 2, 3), 3, 4) 
            else:
                input_part_version = np.swapaxes(np.swapaxes(np.reshape(all_data[data_index  + rotation * len(meshfiles)*N_triangles : data_index + (rotation+1) * len(meshfiles)*N_triangles],
                                     (len(meshfiles),-1,3,13,13)), 2, 3), 3, 4) 
            # timestep x triangular patches x coordinates x 13 x 13

            embedding_part_version = np.reshape(all_emb[data_index  + rotation * len(meshfiles)*N_triangles : data_index + (rotation+1) * len(meshfiles)*N_triangles],
                                                (len(meshfiles),-1,args.hid_rep)) 
            
            
            # move data to range -1,1 for each timestep
            #projected_VV_cc = ( normalize.normalize(projected_VV, ntype='range-0,1') * 2 ) - 1
            # move data to range -1,1 for each timestep
            if 'car' in args.dataset:
                projected_VV_cc = ( normalize.normalize(projected_VV, ntype='range-0,1-mean-0') * 2 ) - 1
            else:
                projected_VV_cc = ( normalize.normalize(projected_VV, ntype='range-0,1') * 2 ) - 1
            
            count_part_patches += len(meshfiles)*N_triangles
            data_index += len(meshfiles)*N_triangles
            
            # copy the patch wise output into this array
            projected_VV_output = np.zeros(projected_VV_cc.shape)
            
            # patch borders appear twice if not boundary. take the average
            appearance_more_than_once = {}
            for VV in range(projected_VV.shape[1]):
                appearance_more_than_once[VV] = []

            for tt in range(len(projected_VV)):
                for pp in range(len(padded_hexagonal_vertex_ids)):
                    if tt == 0:
                        # for first timestep count the appearance for the patch boundaries
                        for vn, vv in enumerate(padded_hexagonal_vertex_ids[pp,patch_indices[:,0],patch_indices[:,1]]):
                            appearance_more_than_once[vv] += [[pp,patch_indices[vn,0],patch_indices[vn,1]]]
                            
                    projected_VV_output[tt,padded_hexagonal_vertex_ids[pp,patch_indices[:,0],patch_indices[:,1]]] = \
                            output_part_version[tt,pp,patch_indices[:,0],patch_indices[:,1]]
                    if len(np.where(input_part_version[tt,pp,patch_indices[:,0],patch_indices[:,1]]==0)[0]) > 0:
                        print(pp)
                        
                # for vertices on the boundary of a patch take the average!
                for vv, matches in appearance_more_than_once.items():
                        if len(matches)>1:
                            projected_VV_output[tt,vv] = np.mean(output_part_version[tt,np.array(matches)[:,0],np.array(matches)[:,1],np.array(matches)[:,2]],axis=0)

            embedding_part_version = np.reshape(embedding_part_version, (len(meshfiles),-1))
            
            # just for doublechecking: reconstruct the input
            projected_VV_input = np.zeros(projected_VV_cc.shape)
            for tt in range(len(projected_VV)):
                for pp in range(len(padded_hexagonal_vertex_ids)):
                    padded_hexagonal_vertex_ids
                    projected_VV_input[tt,padded_hexagonal_vertex_ids[pp,patch_indices[:,0],patch_indices[:,1]]] = \
                            input_part_version[tt,pp,patch_indices[:,0],patch_indices[:,1]]       
                    
            #errors = (projected_VV_input - projected_VV_cc)**2
            #print(np.unique(np.where(errors>0.01)[0]))

            ## calculate the error for each timestep and append to part wise list and list containing all errors
            for tt in range(len(projected_VV)):
                if pname in test_samples or tt in test_tt or version in test_versions:
                    mse2_test.append(np.sum((projected_VV_output[tt] - projected_VV_cc[tt])**2, axis=1) )
                    mse2_total_test.append(np.sum((projected_VV_output[tt] - projected_VV_cc[tt])**2, axis=1) )
                else:
                    mse2.append(np.sum((projected_VV_output[tt] - projected_VV_cc[tt])**2, axis=1) )
                    mse2_total.append(np.sum((projected_VV_output[tt] - projected_VV_cc[tt])**2, axis=1) )
    
    ## part error over all versions
    print('    -> part error <-')
    if len(mse2):
        my_errors = np.asarray(mse2)  # [n_total_graphs, num_nodes]
        mean_error = my_errors.reshape((-1, )).mean(); #mse2=mean_error
        std_error = my_errors.reshape((-1, )).std()
        median_error = np.median(my_errors.reshape((-1, ))) 
        message = 'Train Error: {:.6f}+{:.6f} | {:.6f}'.format(mean_error, std_error, median_error)
    else:
        message = 'Train Error: {:8s}+{:8s} | {:8s}'.format('  --','  --','  --')
    print(message)
    mylogfile.write(message+'\n')

    if len(mse2_test):
        my_errors_test = np.asarray(mse2_test)  # [n_total_graphs, num_nodes]
        mean_error_test = my_errors_test.reshape((-1, )).mean(); #mse2_test=mean_error_test
        std_error_test = my_errors_test.reshape((-1, )).std()
        median_error_test = np.median(my_errors_test.reshape((-1, )))
        message = 'Test  Error: {:.6f}+{:.6f} | {:.6f}'.format(mean_error_test, std_error_test, median_error_test)
    else:
        message = 'Test Error: {:8s}+{:8s} | {:8s}'.format('  --','  --','  --')
    print(message)
    mylogfile.write(message+'\n\n')
    print()
            
    if 'norot' not in args.exp_name: 
        data_index += count_part_patches*2 # the other two rotations


print('\n\nTotal Mean Squared Error:\n') 
mylogfile.write('\n\nTotal Mean Squared Error:\n\n')
mse2_total = [item for sublist in mse2_total for item in sublist] #flatten
if len(mse2_total):
                my_errors = np.asarray(mse2_total) 
                mean_error = my_errors.reshape((-1, )).mean(); #mse2_total=mean_error
                std_error = my_errors.reshape((-1, )).std()
                median_error = np.median(my_errors.reshape((-1, ))) 
                message = 'Train Error: {:.6f}+{:.6f} | {:.6f}'.format(mean_error, std_error, median_error)
else:
                message = 'Train Error: {:8s}+{:8s} | {:8s}'.format('  --','  --','  --')
print(message)
mylogfile.write(message+'\n')

mse2_total_test = [item for sublist in mse2_total_test for item in sublist] #flatten
if len(mse2_total_test):
                my_errors_test = np.asarray(mse2_total_test) 
                mean_error_test = my_errors_test.reshape((-1, )).mean(); #mse2_total_test=mean_error_test
                std_error_test = my_errors_test.reshape((-1, )).std()
                median_error_test = np.median(my_errors_test.reshape((-1, )))
                message = 'Test  Error: {:.6f}+{:.6f} | {:.6f}'.format(mean_error_test, std_error_test, median_error_test)
else:
                message = 'Test Error: {:8s}+{:8s} | {:8s}'.format('  --','  --','  --')
print(message)
mylogfile.write(message+'\n\n')

mylogfile.close()