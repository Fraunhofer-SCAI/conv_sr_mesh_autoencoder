import numpy as np
import sys
sys.path.append('lib')
import torch
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pytorch3d.io import load_obj, save_obj 
from pytorch3d.structures import Meshes
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes


def load_boundary_mesh(center, scale, device, obj_file):
    """ Load the boundary mesh
    input:
    - center: normalization value ( verts = (verts - center)/scale )
    - scale: normalization value ( verts = (verts - center)/scale )
    - device
    - filepath, filename: obj file is at filepath+filename+.obj
    output:
    - boundary_mesh Structure
    """

    verts, faces, aux = load_obj(obj_file)
    
    verts = (verts - center) / scale

    # We construct a Meshes structure   
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])
    
    return boundary_mesh(mesh, device) 

        
def get_boundary_V_E(meshes):
    """  given list of Vertices and Faces, get list of edges and boundary/non-manifold information 
    input:
    - meshes: Meshes Structure
    output:
    - torch vector with indices of boundary vertices
    - 0-1 torch vector with 1 for REAL interior edges
    """
    FF = meshes.faces_packed().detach().cpu().numpy()
    FF_tmp = np.sort(FF, axis=1)
    
    # edges
    edge_1=FF_tmp[:,0:2]
    edge_2=FF_tmp[:,1:]
    edge_3=np.concatenate([FF_tmp[:,:1], FF_tmp[:,-1:]], axis=1)
    EE=np.concatenate([edge_1, edge_2, edge_3], axis=0)
    
    # delete duplicates
    unique_edges_trans, unique_edges_locs, edges_counts=np.unique(EE[:,0]*(10**5)+EE[:,1], 
                                                return_index=True, return_counts=True)
    
    boundary_edges = np.where(edges_counts==1)[0]
    boundary_vertices = np.unique( EE[boundary_edges])
            
    interior_edges = torch.ones(len(meshes.edges_packed()), device=meshes.device)
    interior_edges[boundary_edges] = 0

    return  torch.tensor( boundary_vertices ), interior_edges


def neighbor_stats(F_0, ll_verts, ret_nn=False, output_neighbors=False, printi = False):
    """ Print and output the statistics of the mesh's connectivity. If wanted output connectivity matrix and neighboring faces.
    input:
    - F_0: list of triplets. For each face list the indices of vertices, that define the face
    - ll_verts: number of vertices
    - ret_nn: boolean, output the connectivity statistics (default = False)
    - output_neighbors: boolean, output connectivity information (default = False)
    - printi: boolean, print or not print the statistics (default = False)
    output:
    - if ret_nn and output_neighbors are False: No output 
    - if ret_nn is True: 
      - count_neighbors: connectivity statistics, how often which size of 1-ring neighborhoos
    - if ret_nn is False and output_neighbors is True:
      - get_neighbors: neigboring vertices for each vertex (list of length ll_verts)
      - get_faces: neigboring faces for each vertex (list of length ll_verts)
    """

    if printi:
            print(F_0.shape[0], 'Faces')
            print((ll_verts), 'Vertices')

    get_faces = [set() for ii in range(ll_verts)] # list of sets
    get_neighbors = [set() for ii in range(ll_verts)] # list of sets

    for f_i, trr in enumerate(F_0):
            for jj in trr:
                get_neighbors[jj].update([vv for vv in trr if vv != jj])
                get_faces[jj].add(f_i)

    count_neighbors = [len(get_neighbors[jj]) for jj in range(ll_verts)]
                
    items, occ =  np.unique(count_neighbors, return_counts=True)
    if printi:
        for ii in range(len(items)):
            if items[ii]!=0:
                print(int(items[ii]),':', occ[ii])
    if ret_nn:
        return count_neighbors
    if output_neighbors:
        return get_neighbors, get_faces
    

def order_boundary(boundary_vertices, neighbors, device):
    """ achtung, fehlerhaft
    """

    #vv = boundary_vertices[0]; tmp = [vv] # order the vertices, start with arbitrary one
    #for ww in neighbors[vv]:
    #    if ww in boundary_vertices and ww not in tmp:
    #        tmp += [ww]
    #        vv=ww
    #        break

    #for ii in range(len(boundary_vertices)-2):
    #    for ww in neighbors[vv]:
    #        if ww in boundary_vertices and ww not in tmp:
    #            tmp += [ww]
    #            vv=ww
    #            break
                
    vv = boundary_vertices[7%len(boundary_vertices)]
    tmp = [vv] # order the vertices, start with arbitrary one

    for ii in range(len(boundary_vertices)-1):
        tmp2 = []
        for ww in neighbors[vv]:
            # find neighboring boundary vertices and save to tmp2
            # if corner is defined by only one face, there will be vertices that have three neighboring boundary vertices. 
            # then we select first the neighbor with a smaller neighborhood
            if ww in boundary_vertices and ww not in tmp:
                tmp2 += [ww]
        if len(tmp2) == 1:            
            tmp += [tmp2[0]]
            vv=tmp2[0]
        elif len(tmp2) > 1:
            smallest_nh = 0; nh_size = 10;
            for jj, ww in enumerate(tmp2):
                new_nh_size = len(neighbors[ww])
                if new_nh_size < nh_size:
                    nh_size = new_nh_size
                    smallest_nh = jj
            tmp += [tmp2[smallest_nh]]
            vv=tmp2[smallest_nh]    

    return torch.tensor(tmp, device=device) 

class boundary_mesh:
    
    def __init__(self, mesh, device):
        self.mesh = mesh
        self.device = device
        

        self._src_neighbors = None
        self._boundary_vertices = None
        self._non_boundary_edges = None
        self._real_int_ext_verts = None
        
    def src_neighbors(self):
        """
        Get the list of neighbors for each node.
        Returns:
            list of neighbors for each node (V, xx).
        """
        if self._src_neighbors is None:
            
            self._src_neighbors, _ = self.neighbor_stats(output_neighbors=True, subsampled=False, printi=False)

        return self._src_neighbors           
    
    def real_int_ext_verts(self):
        """ return 0-1 tensor with 1 for REAL interior vertices (those vertices that are not boundary vertices)
        """
        if self._real_int_ext_verts is None:
            self._real_int_ext_verts =  torch.ones(len(self.mesh.verts_packed()), device=self.device)
            self._real_int_ext_verts[self.boundary_vertices()] = 0
            
        return self._real_int_ext_verts
    
    def boundary_vertices(self):
        """ return tensor with indices of boundary vertices
        """
        if self._boundary_vertices is None:
            self._boundary_vertices, self._non_boundary_edges = get_boundary_V_E(self.mesh)
            #self._boundary_vertices = order_boundary(self._boundary_vertices.cpu().detach().numpy(), self.src_neighbors(), self.device)    
        
        return self._boundary_vertices
    
    def non_boundary_edges(self):
        """ return 0-1 tensor with 1 for REAL interior edges (those edges that are not boundary edges)
        """        
        if self._non_boundary_edges is None:
            self._boundary_vertices, self._non_boundary_edges = get_boundary_V_E(self.mesh)
            #self._boundary_vertices = order_boundary(self._boundary_vertices.cpu().detach().numpy(), self.src_neighbors(), self.device)
        
        return self._non_boundary_edges
    
    
    def neighbor_stats(self, ret_nn=False, output_neighbors=False, printi = False):
        """ Print and output the statistics of the mesh's connectivity. If wanted output connectivity matrix and neighboring faces.
        input:
        - ret_nn: boolean, output the connectivity statistics (default = False)
        - output_neighbors: boolean, output connectivity information (default = False)
        - printi: boolean, print or not print the statistics (default = False)
        output:
        - if ret_nn and output_neighbors are False: No output 
        - if ret_nn is True: 
          - count_neighbors: connectivity statistics, how often which size of 1-ring neighborhoos
        - if ret_nn is False and output_neighbors is True:
          - get_neighbors: neigboring vertices for each vertex (list of length ll_verts)
          - get_faces: neigboring faces for each vertex (list of length ll_verts)
        """
        F_0 = self.mesh.faces_padded()[0].cpu().numpy()
        ll_verts = len(self.mesh.verts_padded()[0])
        
        return neighbor_stats(F_0, ll_verts, ret_nn=ret_nn, output_neighbors=output_neighbors, printi = printi)
        
        
    def subdivideMesh(self):
        """ subdivide the subsampled_mesh
        """

        subdivide = SubdivideMeshes()
        mesh = subdivide(self.mesh)
        verts = mesh.verts_list()[0]

        self.mesh = mesh
        self._src_neighbors = None
        self._boundary_vertices = None
        self._non_boundary_edges = None
        self._real_int_ext_verts = None

    def save_mesh(self, center, scale, filepath='my_data/mesh/', filename='part'):

        # Fetch the verts and faces of the final predicted mesh
        final_verts, final_faces = self.mesh.get_mesh_verts_faces(0)

        # Scale normalize back to the original target size
        final_verts = final_verts * scale + center

        # Store the predicted mesh using save_obj
        final_obj = os.path.join('./', filepath, filename+'.obj')
        save_obj(final_obj, final_verts, final_faces)        

                   
                   

    def plot_3Dmesh(self, pl_normal_ls = False, pl_edge_ls = False, second_mesh = None, second_bound = None, save_fig=None, title=None):

        VV = self.mesh.verts_padded()[0].cpu().detach().numpy()
        FF = self.mesh.faces_padded()[0].cpu().detach().numpy()
        EE = self.mesh.edges_packed().cpu().detach().numpy()
        
        if second_mesh is not None:
            VV_2 = second_mesh.verts_padded()[0].cpu().detach().numpy()
            EE_2 = second_mesh.edges_packed().cpu().detach().numpy()

        boundary_vertices_np = self.boundary_vertices().cpu().detach().numpy()

        mesh2 = [[VV[ttr_ii] for ttr_ii in ttr] for ttr in FF]

        if pl_edge_ls == True or pl_normal_ls == True:
            loss_edge_per_edge_np = my_mesh_edge_loss(self.mesh, meancalc=False).cpu().detach().numpy()
            loss_normal_per_edge_np = my_mesh_normal_consistency(self.mesh, meancalc=False).cpu().detach().numpy()

            max_nl = np.max(loss_normal_per_edge_np)
            min_nl = np.min(loss_normal_per_edge_np)

            final_loss_normal = (loss_normal_per_edge_np-min_nl)*1/(max_nl-min_nl)

            edges_packed = self.mesh.edges_packed()

            print(len(self.mesh.verts_packed()))
            print(np.shape(VV))

            edge_loss_at_vert = [ [] for ii in np.zeros(len(self.mesh.verts_packed()))]
            normal_loss_at_vert = [ [] for ii in np.zeros(len(self.mesh.verts_packed()))]

            for ie, ee in enumerate(edges_packed):
                v1 = ee[0]; v2 = ee[1]
                if v2 in self.boundary_vertices():
                    edge_loss_at_vert[v2] += [loss_edge_per_edge_np[ie]]
                    normal_loss_at_vert[v2] += [loss_normal_per_edge_np[ie]]
                if v1 in self.boundary_vertices():
                    edge_loss_at_vert[v1] += [loss_edge_per_edge_np[ie]]
                    normal_loss_at_vert[v1] += [loss_normal_per_edge_np[ie]]

            print(np.shape(boundary_vertices_np))        
            print(len(self.boundary_vertices())) 

            #edge_loss_boundary = np.zeros(len(self.boundary_vertices()))
            normal_loss_boundary = np.zeros(len(self.boundary_vertices()))

            for vi, vv in enumerate(self.boundary_vertices()):
                #edge_loss_boundary[vi] = np.mean(edge_loss_at_vert[vv])
                normal_loss_boundary[vi] = np.mean(normal_loss_at_vert[vv])

            print('Plot normal loss.')
            loss_boundary = normal_loss_boundary
        else:
            final_loss_normal = np.ones(len(EE))
            loss_boundary = np.ones(len(self.boundary_vertices()))

        VV_select = np.asarray([VV[ii] for ii in range(len(VV))])

        fig = plt.figure(figsize=(18,15))
        plt.axis('off')
        
        if title is not None:
            plt.title(title)

        for ii in range(2):
            ax = fig.add_subplot(2,1,ii+1, projection='3d')
            ax.scatter3D(VV_select[:,0], VV_select[:,2], -VV_select[:,1],alpha=0)
            #ax.axis('equal')
            ax.axis('off')

            for eind, ee in enumerate(EE):
                if self.non_boundary_edges()[eind] == 0: # boundary edges
                    ax.plot([VV[ee[0]][0],VV[ee[1]][0]],[VV[ee[0]][2],VV[ee[1]][2]],[-VV[ee[0]][1],-VV[ee[1]][1]], 
                            color = cm.viridis(final_loss_normal[eind]), alpha=1, lw=3, zorder=2)
                else:
                    ax.plot([VV[ee[0]][0],VV[ee[1]][0]],[VV[ee[0]][2],VV[ee[1]][2]],[-VV[ee[0]][1],-VV[ee[1]][1]], 
                            color ='black', alpha=0.5, lw=1, zorder=2)
            #ax.scatter3D(VV[:,0], VV[:,1], VV[:,2],c=color_vert,alpha=1)
            #ax.scatter3D(VV_select[:,0], VV_select[:,1], VV_select[:,2],alpha=0.4)

            if second_mesh is not None:
                #for eind, ee in enumerate(EE_2):
                #    ax.plot([VV_2[ee[0]][0],VV_2[ee[1]][0]],[VV_2[ee[0]][1],VV_2[ee[1]][1]],[VV_2[ee[0]][2],VV_2[ee[1]][2]], 
                #            color ='navy', alpha=0.3, lw=0.5, zorder=3)
                if second_bound is not None:
                    for bbii in range(len(second_bound)):
                        b_edge = second_bound[[bbii,(bbii+1)%len(second_bound)]] # this only works if boundary is sorted
                        ax.plot(VV_2[b_edge][:,0], VV_2[b_edge][:,2], -VV_2[b_edge][:,1], 
                                color ='red', alpha=0.5, ls = '-', zorder=3)


            ax.set_xlabel('X')
            #ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(190, 105 + ii*190)
            

            
        if save_fig is not None:
            plt.savefig(save_fig)