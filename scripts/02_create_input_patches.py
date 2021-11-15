import numpy as np
import os
import os.path as osp
import pickle
import argparse

import igl

from mesh import clean_mesh_get_edges
from mesh import subdivided_mesh, copy_values_to_hextria_over_time

from utils import utils, normalize

import time

parser = argparse.ArgumentParser(description='creation of input data')
parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment.')
parser.add_argument('--dataset', type=str, default='gallop', help='Name of the dataset.')
parser.add_argument('--device_idx', type=str, default='cpu', help='Device. Can be CPU or the id of the GPU. This script runs always on CPU.')

# mesh refinement
parser.add_argument('--refine', type=int, default=3, help='Level of refinement. Tested for 3.')

# patch arguments
parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size of the first convolutional layer. Tested for 2.') # this is also the size of the padding
parser.add_argument('--rotation_augment', type=int, default=1, help='Rotate the patches by 60 and 120 degrees to augment the size of the training set.')

# data split
parser.add_argument('--test_split', nargs="+", type=str, default=['elephant'], help='List of test samples. For those the train-test-split is 0-100.') #type=str, default='elephant') # train-test-split: 0-100 for test samples
parser.add_argument('--test_version', nargs="+", type=str, default=[], help='Only for car crash. versions that are completly part of test data.') # only for car crash
parser.add_argument('--test_ratio', type=float, default=0.25, help='train-test-split. Default: 75% training, 25% testing plus the test samples given by variable test_split.') # train-test-split: 75-25 for train samples

args = parser.parse_args()

# features
# absolute values
# selected_dims = [0,1,2]

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.data_raw = osp.join(args.data_fp, 'raw')
args.data_preprocessed = osp.join(args.data_fp, 'preprocessed')
args.data_semireg = osp.join(args.data_fp, 'semiregular')
if args.rotation_augment == 0:
    args.rotation_augment = False
    args.data_train_patches = osp.join(args.data_fp, 'train_patches_{}_norot'.format(args.exp_name))
else:
    args.data_train_patches = osp.join(args.data_fp, 'train_patches_{}'.format(args.exp_name))
utils.mkdirs(args.data_train_patches)

semireg_logs = osp.join(args.data_semireg, 'logs')
utils.mkdirs(semireg_logs)

# write arguments to a log file
with open(semireg_logs+'/arguments_02_input_patches_{}_{}.txt'.format(args.dataset, args.exp_name), 'w') as f:
    for aa in list(args.__dict__.keys()):
        if args.work_dir in str(args.__dict__[aa]):
            text = '(path) {}: {}\n'.format(aa, args.__dict__[aa])
        else:
            text = '--{} {}\n'.format(aa, args.__dict__[aa])
        f.write(text)
        print(text, end='')

mylogfile = open(semireg_logs+'/arguments_02_input_patches_{}_{}.txt'.format(args.dataset, args.exp_name), "a")
mylogfile.write("\n-----------------\nLOG \n-----------------\n")
        

## versions
versions = [f.name for f in os.scandir(args.data_raw) if f.is_dir() and 'checkpoints' not in f.name]
print('Versions:', versions)

test_versions = []  ## add posibly versions to this list if wanted
if len(args.test_version) > 0:
    test_versions = args.test_version #['sim_041','sim_049']
    print('Test Versions:', test_versions)
    mylogfile.write("Test Versions: {}\n".format(test_versions))
    
    
## parts/samples (for every version the same!)
samples = [f.name for f in os.scandir(osp.join(args.data_raw, versions[0])) if f.is_dir() and 'checkpoints' not in f.name]
print('Samples:', samples)
test_samples = [sa for sa in samples if sa in args.test_split]

print('\nTest Sample:', test_samples)
train_samples = [sa for sa in samples if sa  not in test_samples] 
print('Train Samples:', train_samples, '\n')


# if refine = 3
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
loss_mask_refine3 = np.zeros((1,1,13,13))
loss_mask_refine3[:,:,patch_indices[:,0],patch_indices[:,1]]=1

np.random.seed(1)

data_index = 0
padded_hexagonal_vertex_ids_per_part= []
vertex_ids_per_hexagonal_patch_per_part= []

if args.rotation_augment == False:
    all_rotations = [0]
else:
    all_rotations = [0,1,2]

for pp, pname in enumerate(train_samples+test_samples):
    
    print('################')
    print('Sample', pname)
    mylogfile.write('\n\n#########\nSample {}'.format(pname))
    
    data_preprocessed_part = osp.join(args.data_preprocessed, pname)
    tmp_exp_name = args.exp_name
    if 'inter' in args.exp_name:
        tmp_exp_name = args.exp_name[:-6]
    base_part = os.path.join(data_preprocessed_part, "mesh_{}_remesh_exp_{}.obj".format(pname, tmp_exp_name))
    data_semireg_part = osp.join(args.data_semireg, pname)
    
    ##### Load Base Data    
    VV_base, _, _, FF_base, _, _ = igl.read_obj(base_part) 
    _, FF_base, EE_base, boundary_edges, boundary_vertices, nonmanifold_edges = clean_mesh_get_edges(VV_base, FF_base) # delete dublicate faces if there are
    
    N_triangles = len(FF_base)
    print('  Base faces: {}'.format(N_triangles))
    mylogfile.write('\n  Base faces: {}'.format(N_triangles))
    
    #### Create Datastructure
    submeshfilename = os.path.join(data_semireg_part,'subdivided_mesh_{}_{}.p'.format(pname, tmp_exp_name))
    if pp>0 and 'FAUST' in args.dataset:
        print('  Use subdivision from first sample.') # this only works for faust, because meshes are in correspondence. speeds up the preprocessing
    elif osp.isfile(submeshfilename):
        print('  Load saved subdivision {}'.format(submeshfilename))
        with open(submeshfilename, 'rb') as subdivided_mesh_file:
            sub_mesh = pickle.load(subdivided_mesh_file)
    else:
        print('  ',end='')
        start_time_sub = time.time()
        sub_mesh = subdivided_mesh(VV_base, EE_base, FF_base, level=args.refine)
        mylogfile.write("  Subdivide faces of triangle mesh {} times ({} seconds)\n ".format(args.refine, np.round(time.time()-start_time_sub,2)))
        
        with open(submeshfilename, 'wb') as subdivided_mesh_file:
            pickle.dump(sub_mesh, subdivided_mesh_file)
    
    for rotation in all_rotations:
        
        print('  Rotation:',rotation)
        # Hex-representation and padding
        list_padded_hexagonal = sub_mesh.create_final_padded_hextrias(kernel_size=args.kernel_size, rotation=rotation)
        
        if rotation==0:
            padded_hexagonal_vertex_ids_per_part += [np.array(list_padded_hexagonal,dtype=int)]
            
            
        #### Load Mesh data at vertices

        height_padded, width_padded = list_padded_hexagonal[0].shape[2:]

        if pp == 0 and rotation==0:
            index_train = np.zeros((0))
            index_test = np.zeros((0))
            
            data_all = np.zeros((0, 3, height_padded, width_padded))
            
    
        for kk,version in enumerate(versions):

            pickle_data = os.path.join(data_semireg_part, 
                                "projected_mesh_{}_{}_remesh_exp_{}_refinelevel_{}_vertex_values.p".format(pname, 
                                        version, tmp_exp_name, args.refine))
            
            if rotation == 0:
                print('    Version {}: '.format(version), end='')
                print(pickle_data[len(data_semireg_part)+1:])
                
            # load the data
            with open(pickle_data, "rb") as file:
                projected_VV = pickle.load(file) # (timesteps, vertices, 3)
                
            # move data to range -1,1 for each timestep
            if 'car' in args.dataset:
                projected_VV_cc = ( normalize.normalize(projected_VV, ntype='range-0,1-mean-0') * 2 ) - 1
            else:
                projected_VV_cc = ( normalize.normalize(projected_VV, ntype='range-0,1') * 2 ) - 1
            
            if rotation == 0:
                print('      Minimum Data Values:', np.min(projected_VV_cc, axis=(0,1)),end=',')
                print(' Maximum Data Values:', np.max(projected_VV_cc, axis=(0,1)))
            
            if 'car' not in args.dataset:
                meshfiles = [f.name for f in os.scandir(osp.join(args.data_raw, version, pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
                meshfiles.sort()

                select_tt = np.arange(0,len(meshfiles),1)

                if args.test_ratio > 0:
                    test_tt = select_tt[-int(len(meshfiles)* args.test_ratio ):] # use the last int(len(meshfiles)* args.test_ratio ) timesteps for testing
                else:
                    test_tt = []
            
            else:
                # raw data is not uploaded yet. extract the number of timesteps from the projected vertices.
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
                        mylogfile.write('    Random test timestps: {}\n'.format(test_tt))
                else:
                    test_tt = []
                    
            
            data_features_tmp = np.zeros((len(select_tt),N_triangles,3,height_padded,width_padded))
            data_features_tmp = copy_values_to_hextria_over_time(data_features_tmp, projected_VV_cc, list_padded_hexagonal, kernel_size=args.kernel_size, average_padding=True)
                        
            data_all = np.append(data_all, np.reshape(data_features_tmp, (-1,3,height_padded,width_padded) ), axis=0)
                    
            for tt in select_tt:
                if rotation==0:
                    if tt == select_tt[0]:
                        print('      Time', tt, end='')
                    elif tt == select_tt[1]:
                        print(', {}'.format(tt), end='')
                    elif tt == select_tt[-1]:
                        print(', ..., {}'.format(tt), end='')

                index_tmp = np.arange(data_index, data_index+N_triangles)
                data_index += N_triangles

                if pname in test_samples or version in test_versions:   
                    index_test = np.append(index_test, index_tmp, axis=0)
                else:
                    if tt in test_tt:
                        index_test = np.append(index_test, index_tmp, axis=0)
                    else:
                        index_train = np.append(index_train, index_tmp, axis=0)

            if rotation==0:
                print('')

        
index_train = np.asarray(index_train, dtype=int)
index_test = np.asarray(index_test, dtype=int)

print()
print()
print(len(index_train),'train samples.')
print('Shape of the train data',data_all[index_train].shape)
print()
print(len(index_test),'test samples.')
print('Shape of the test data',data_all[index_test].shape)

mylogfile.write('\n\n#########\n{} train samples.'.format(len(index_train)))
mylogfile.write('\nShape of the train data {}'.format(data_all[index_train].shape))
mylogfile.write('\n\n{} test samples.'.format(len(index_test)))
mylogfile.write('\nShape of the test data {}'.format(data_all[index_test].shape))
mylogfile.write("\n\nSave the patches to: {}".format(args.data_train_patches[len(args.data_fp)+1:]))

### Save the patches
print("\nSave the patches to:", args.data_train_patches[len(args.data_fp)+1:])
with open(args.data_train_patches+'/all_data.npy', 'wb') as f:
    np.save(f, data_all)
with open(args.data_train_patches+'/train_index.npy', 'wb') as f:
    np.save(f, index_train)
with open(args.data_train_patches+'/test_index.npy', 'wb') as f:
    np.save(f, index_test)
for pp, pname in enumerate(train_samples+test_samples):
    with open(args.data_train_patches+'/padded_hexagonal_vertex_ids_sample_{}.npy'.format(pname), 'wb') as f:
        np.save(f, padded_hexagonal_vertex_ids_per_part[pp])
        
mylogfile.close()