import numpy as np
import os
import os.path as osp
import pickle
import time

from utils import utils, read_plymesh, barycentric
from mesh import get_edges, sort_connected_vertices, clean_mesh_get_edges

import torch
import torch.backends.cudnn as cudnn
from pytorch3d.io import load_obj, save_obj  
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points
import igl

import argparse

import igl

parser = argparse.ArgumentParser(description='mesh preprocessing and projection over time')
parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment.')
parser.add_argument('--dataset', type=str, default='gallop', help='Name of the dataset.')
parser.add_argument('--device_idx', type=str, default='cpu', help='Device. Can be CPU or the id of the GPU. This script runs always on CPU.')

# mesh refinement
parser.add_argument('--refine', type=int, default=3, help='Level of refinement. Tested for 3.')

# mesh register
parser.add_argument('--registerneighbors', type=int, default=3, help='Vertices to consider for parametization of remeshing result. 1: nearest neighbor. 3: barycentrc coordinates of vertices of closest face.') #3: barycentric coordinates

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.data_raw = osp.join(args.data_fp, 'raw')
args.data_preprocessed = osp.join(args.data_fp, 'preprocessed')
args.data_semireg = osp.join(args.data_fp, 'semiregular')

utils.mkdirs(args.data_preprocessed)
utils.mkdirs(args.data_semireg)
semireg_logs = osp.join(args.data_semireg, 'logs')
utils.mkdirs(semireg_logs)

# write arguments to a log file
with open(semireg_logs+'/arguments_01_preprocessing_{}.txt'.format(args.dataset), 'w') as f:
    for aa in list(args.__dict__.keys()):
        if args.work_dir in str(args.__dict__[aa]):
            text = '(path) {}: {}\n'.format(aa, args.__dict__[aa])
        else:
            text = '--{} {}\n'.format(aa, args.__dict__[aa])
        f.write(text)
        print(text, end='')

mylogfile = open(semireg_logs+'/arguments_01_preprocessing_{}.txt'.format(args.dataset), "a")
mylogfile.write("\n-----------------\nLOG \n-----------------\n")
        
## versions
versions = [f.name for f in os.scandir(args.data_raw) if f.is_dir() and 'checkpoints' not in f.name]
print('Versions:', versions)
mylogfile.write('\nVersions: {}\n'.format(versions))

## parts/samples (for every version the same!)
samples = [f.name for f in os.scandir(osp.join(args.data_raw, versions[0])) if f.is_dir() and 'checkpoints' not in f.name]
samples.sort()
print('Samples:', samples)
mylogfile.write('Samples: {}\n'.format(samples))

for kk,version in enumerate(versions):
    
    for pp, pname in enumerate(samples):
        
        print('\nPreprocess sample ', pname, sep='')
        mylogfile.write('\nPreprocess sample {}\n'.format(pname))
        
        meshfiles = [f.name for f in os.scandir(osp.join(args.data_raw, version, pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
        meshfiles.sort() # sort by timestep if the file naming is done nicely
        
        #print(meshfiles)
        
        if 'FAUST' in args.dataset:
            reference = [f.name for f in os.scandir(osp.join(args.data_raw)) if f.is_file() and '.obj' in f.name and 'reference' in f.name][0]
        else:
            reference = [f.name for f in os.scandir(osp.join(args.data_raw, version, pname)) if f.is_file() and '.obj' in f.name and 'reference' in f.name][0]
        
        if 'FAUST' in args.dataset:
            verts_np, _, _, faces_np, _, _ = igl.read_obj( osp.join(args.data_raw, reference) )
        else:
            verts_np, _, _, faces_np, _, _ = igl.read_obj( osp.join(args.data_raw, version, pname, reference) )
            

        print('  Vertices: {}; Faces: {}'.format(verts_np.shape,faces_np.shape))
                
        data_preprocessed_part = osp.join(args.data_preprocessed, pname)
        utils.mkdirs(data_preprocessed_part)
            
        if kk == 0:
            print('  Save reference mesh for {}.'.format(pname))

            file_mesh = os.path.join(data_preprocessed_part, "mesh_{}_reference.obj".format(pname) )
            ret = igl.write_obj(file_mesh, verts_np, faces_np)
            
        # timesteps
        tt = range(0,len(meshfiles),1)
        
        print('  {} Timesteps'.format(len(tt)))
        mylogfile.write('  {} Timesteps\n'.format(len(tt)))

        # vertices over time are saved to a pickle document to allow faster processing
        vertices_over_time = np.zeros((verts_np.shape[0], len(tt), 3))
          
        for tn, t in enumerate(tt):
            obj_file = osp.join(args.data_raw, version, pname, meshfiles[tn])
            if '.obj' in obj_file:
                verts_np, _, _, faces_np, _, _ = igl.read_obj( obj_file )
            elif '.ply' in obj_file:
                verts_np, faces_np = read_plymesh.read_plymesh( obj_file )
            vertices_over_time[:,tn] = verts_np
        file_mesh = os.path.join(data_preprocessed_part, "mesh_{}_{}_vertex_values.p".format(pname, version))
        print('  Save Vertices over time (size :{}) to {}.'.format(vertices_over_time.shape, file_mesh[len(data_preprocessed_part)+1:]))
        with open(file_mesh, "wb") as file:
            pickle.dump( vertices_over_time, file )
          
        if kk == 0:

            file_mesh = os.path.join(data_preprocessed_part, "boundary_{}.txt".format(pname))
            print('  Save boundary to {}.'.format(file_mesh[len(data_preprocessed_part)+1:]))
            # boundary
            # for the gallop and FAUST collection the meshes do not have a boundary
            open(file_mesh, 'a').close()


            if ('FAUST' in args.dataset) and pp != 0:
                print('  Use simplification from part 0.')
                file_mesh = os.path.join(data_preprocessed_part, "mesh_{}_remesh_exp_{}.obj".format(pname, args.exp_name))
                file_mesh_p0 = os.path.join(args.data_preprocessed, samples[0], "mesh_{}_remesh_exp_{}.obj".format(samples[0], args.exp_name))
                print("    cp {} {}".format(file_mesh_p0, file_mesh))
                os.system("cp {} {}".format(file_mesh_p0, file_mesh))

            else:
                # remeshing!!
                reference_mesh = os.path.join(data_preprocessed_part, "mesh_{}_reference.obj".format(pname) )
                file_mesh = os.path.join(data_preprocessed_part, "mesh_{}_remesh_exp_{}.obj".format(pname, args.exp_name))
                print('  Remesh Result: mesh_{}_remesh_exp_{}.obj'.format(pname, args.exp_name))

                
                
                
mylogfile.write('\n\n------------------\nProjection over time\n')                
             
    
nn_coords_per_part = []

for kk,version in enumerate(versions):
    print('\n#####\nVersion: ', version, sep='')
    mylogfile.write('\n\n#####\nVersion: {}\n'.format(version))
    
    for pp, pname in enumerate(samples):

        data_preprocessed_part = osp.join(args.data_preprocessed, pname)
        data_semireg_part = osp.join(args.data_semireg, pname)
        
        meshfiles = [f.name for f in os.scandir(osp.join(args.data_raw, version, pname)) if f.is_file() and ('.obj' in f.name or '.ply' in f.name) and 'reference' not in f.name]
        meshfiles.sort() # sort by timestep if the file naming is done nicely
        
        if kk == 0:
            print('\nBarycentric coordinates for sample ', pname, sep='')
            
            if 'FAUST' in args.dataset:
                reference = [f.name for f in os.scandir(osp.join(args.data_raw)) if f.is_file() and '.obj' in f.name and 'reference' in f.name][0]
            else:
                reference = [f.name for f in os.scandir(osp.join(args.data_raw, version, pname)) if f.is_file() and '.obj' in f.name and 'reference' in f.name][0]

            if 'FAUST' in args.dataset:
                VV_target, _, _, FF_target, _, _ = igl.read_obj( osp.join(args.data_raw, reference) )
            else:
                VV_target, _, _, FF_target, _, _ = igl.read_obj( osp.join(args.data_raw, version, pname, reference) )

            EE_target, _, _, _ = get_edges(VV_target, FF_target)        

            VV_remesh, _, _, FF_remesh, _, _ = igl.read_obj(data_semireg_part+'/'+'{}_remesh_exp_{}_refinelevel_{}.obj'.format(pname,args.exp_name,args.refine)) 

            _, FF_remesh, EE_remesh, boundary_edges, boundary_vertices, nonmanifold_edges = clean_mesh_get_edges(VV_remesh, FF_remesh)

            
            if args.registerneighbors == 1:
                VV_target_tensor = torch.tensor(np.reshape(VV_target,
                                        (1,VV_target.shape[0],VV_target.shape[1]))).float()
                VV_remesh_tensor = torch.tensor(np.reshape(VV_remesh,
                                        (1,VV_remesh.shape[0],VV_remesh.shape[1]))).float() 
                # find closest neighbor
                knn = knn_points(VV_remesh_tensor, 
                                VV_target_tensor, 
                                lengths1=None, lengths2=None, K=1 ) 
                nn_idx = knn.idx[0].cpu().numpy()
                # set corresponding factors to 1
                nn_coords = np.ones((len(nn_idx),len(VV_target)))  # len(VV_remesh) x len(VV_target)

            elif args.registerneighbors == 3:

                nn_coords = np.zeros((len(VV_remesh),len(VV_target)))

                for vv in range(len(VV_remesh)):

                    # find the closest face
                    closest_face = np.argmin(np.sum((VV_target[FF_target] - VV_remesh[vv])**2, 
                                                            axis=(1,2))) 

                    # calculate the barycentric coordinates of the projection of VV_remesh[vv] to
                    # the plane defined by the closest traingular face
                    
                    closest_face_vids = np.sort(FF_target[closest_face])                    
                    nn_coords[vv,closest_face_vids] = np.array(barycentric.Barycentric(VV_remesh[vv], 
                                                          VV_target[closest_face_vids[0]], 
                                                          VV_target[closest_face_vids[1]], 
                                                          VV_target[closest_face_vids[2]]))

                    # move the projection inside the triangle (to the boundary)
                    # count the times the point is not inside the triangle
                    if np.sum(np.abs(nn_coords[vv]))>1:
                        nn_coords[vv,closest_face_vids] = barycentric.move_projection_inside_triangle(nn_coords[vv,closest_face_vids])
                    
                ## if remeshed mesh vertices should be saved for the reference mesh
                VV_remesh_proj = np.zeros(VV_remesh.shape)
                VV_remesh_proj = (np.transpose(np.dot(np.transpose(VV_target), 
                                            np.transpose(nn_coords))) )
                
                        
            nn_coords_per_part += [nn_coords]
        else:
            nn_coords = nn_coords_per_part[pp]
            
            
        # timesteps
        tt = range(0,len(meshfiles),1)
        projected_VV = np.zeros((len(meshfiles), len(nn_coords), 3))
        
        file_mesh = os.path.join(data_preprocessed_part, "mesh_{}_{}_vertex_values.p".format(pname, version))
        print('Load Vertices over time for part {}. Shape: '.format(pname), end='')
        with open(file_mesh, "rb") as file:
            vertices_over_time = pickle.load(file )
        vertices_over_time = np.swapaxes(vertices_over_time, 0, 1)
        print(vertices_over_time.shape)
        mylogfile.write('\nLoad Vertices over time for part {}. Shape: {}'.format(pname, vertices_over_time.shape)) 

        if args.registerneighbors == 1:
            projected_VV = vertices_over_time[:,nn_idx[:,0]]

        else:
            for time in range(vertices_over_time.shape[0]):
                projected_VV[time] = (np.transpose(np.dot(np.transpose(vertices_over_time[time]), 
                                            np.transpose(nn_coords))) )
        
        print('  Projected shape:',projected_VV.shape, end='.')
        mylogfile.write('\n  Projected shape: {}'.format(projected_VV.shape))


        #### save
        file_mesh = os.path.join(data_semireg_part, "projected_mesh_{}_{}_remesh_exp_{}_refinelevel_{}_vertex_values.p".format(pname, version, args.exp_name,args.refine))
        with open(file_mesh, "wb") as file:
            pickle.dump(projected_VV, file)
        print(' Save projection to:',file_mesh[len(data_semireg_part)+1:])
        mylogfile.write('\n  Save projection to: {}'.format(file_mesh[len(data_semireg_part)+1:])          )
          

mylogfile.close()


