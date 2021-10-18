import torch
import numpy as np
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes

from hexagdly.hexagdly_tools import plot_hextensor

from mesh import get_edges, connectivity_list, sort_connected_vertices, get_neigbors_in_hex

import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap('viridis')


def get_boundary_verts(EE, FF):
    """ Get boundary vertices given edges and faces
    # input:
    # - EE: edges, numpy array (E,2)
    # - FF: faces, numpy array (F,3)
    # output:
    # - array with indices of boundary vertices
    """
    _, _, bb, _ = get_edges(0, FF)

    return  bb

def copy_values_to_hextria(hex_data_feature, projected_VV_cc, list_padded_hexagonal, kernel_size=2, average_padding=True):
    """ copy the vertex data to the hextria matrix
    input:
    - hex_data: shape (N_trianlges, 1, height, weight), indices of the vertices whose data will be copied
    output:
    - hex_data_feature: shape (N_trianlges, projected_VV_cc.shape[1], height, weight)
    """
    
    height_padded = hex_data_feature.shape[2]
    width_padded = hex_data_feature.shape[3]
    
    if hex_data_feature.shape[1] != projected_VV_cc.shape[1]:
        print('feature dimension is unequal to hexagonal data feature vector.')
        return None
        
    for ff_i in range(hex_data_feature.shape[0]):

        for hh in range(height_padded):
            for ww in range(width_padded):
                vx = int(list_padded_hexagonal[ff_i][0,0,hh,ww])
                if vx >= 0:
                    hex_data_feature[ff_i,:,hh,ww] = projected_VV_cc[vx]

        if average_padding:
            #fill up the empty padding by the average of the neighboring nodes so that the network doesnt learn the shape of the padding 
            for kk in range(kernel_size):
                for hh in range(height_padded):
                    for ww in range(width_padded):
                        vx = int(list_padded_hexagonal[ff_i][0,0,hh,ww])
                        if vx == -(kk+1):
                            neighbors_hex = np.array([ hex_data_feature [ff_i,:,nn[0],nn[1]] for nn in get_neigbors_in_hex(hh, ww, height_padded-1, width_padded-1) if any(hex_data_feature[ff_i,0,nn[0],nn[1]] != np.zeros((projected_VV_cc.shape[1])))])
                            if len(neighbors_hex):
                                value_avg = np.sum(neighbors_hex, axis=0)/len(neighbors_hex)
                                hex_data_feature [ff_i,:,hh,ww] = value_avg
    return hex_data_feature

def copy_values_to_hextria_over_time(hex_data_feature, projected_VV_cc, list_padded_hexagonal, kernel_size=2, average_padding=True):
    """ copy the vertex data to the hextria matrix
    input:
    - hex_data_feature: shape (projected_VV_cc.shape[0], N_trianlges, projected_VV_cc.shape[2], height, weight)
    - projected_VV_cc: shape (#timesteps , #vertices , 3), vertex values
    - list_padded_hexagonal: index of vertices for each patch
    output:
    - hex_data_feature: shape (projected_VV_cc.shape[0], N_trianlges, projected_VV_cc.shape[2], height, weight)
    """
    
    height_padded = hex_data_feature.shape[3]
    width_padded = hex_data_feature.shape[4]
    
    if hex_data_feature.shape[2] != projected_VV_cc.shape[2]:
        print('feature dimension is unequal to hexagonal data feature vector.')
        return None
        
    for ff_i in range(hex_data_feature.shape[1]):

        for hh in range(height_padded):
            for ww in range(width_padded):
                vx = int(list_padded_hexagonal[ff_i][0,0,hh,ww])
                if vx >= 0:
                    hex_data_feature[:,ff_i,:,hh,ww] = projected_VV_cc[:,vx]

        if average_padding:
            #fill up the empty padding by the average of the neighboring nodes so that the network doesnt learn the shape of the padding 
            for kk in range(kernel_size):
                for hh in range(height_padded):
                    for ww in range(width_padded):
                        vx = int(list_padded_hexagonal[ff_i][0,0,hh,ww])
                        if vx == -(kk+1):
                            for tt in range(hex_data_feature.shape[0]):
                                neighbors_hex = np.array([ hex_data_feature [tt,ff_i,:,nn[0],nn[1]] for nn in get_neigbors_in_hex(hh, ww, height_padded-1, width_padded-1) if any(hex_data_feature[tt,ff_i,0,nn[0],nn[1]] != np.zeros((projected_VV_cc.shape[2])))])
                                if len(neighbors_hex):
                                    value_avg = np.sum(neighbors_hex, axis=0)/len(neighbors_hex)
                                    hex_data_feature [tt,ff_i,:,hh,ww] = value_avg
    return hex_data_feature


def plot_hextrias(hex_data, hex_label=None, title='Triangular patches', save_fig=''):
    height_padded = hex_data.shape[2]
    width_padded = hex_data.shape[3]
            
    # for each feature dimension
    for ii in range(hex_data.shape[1]):        
        fig = plt.figure(figsize=(7,7))
        plt.title(title+', feature-dim {}'.format(ii))
        plt.axis('off')

        plot_rows = np.ceil(np.sqrt(hex_data.shape[0]))
        
        values_plot_min = np.min(hex_data[:,ii])
        values_plot_diff = np.max(hex_data[:,ii]) - values_plot_min

        for ff_i in range(hex_data.shape[0]):
            ax = fig.add_subplot(int(plot_rows),int(hex_data.shape[0]/plot_rows)+1,ff_i+1)

            #ax.axis('equal')
            ax.axis('off')
            
            for hh in range(height_padded):
                for ww in range(width_padded):
                    #print(VV2[vv,0], VV2[vv,1])
                    if hex_data [ff_i,0,hh,ww] != 0:
                        ax.scatter(ww , hh + 0.5 * (ww%2), c = cmap(np.array([ (hex_data [ff_i,ii,hh,ww] - values_plot_min) / values_plot_diff ])))

            if hex_label is not None:
                ax.text(width_padded/2, 4*height_padded/5, 'label ' + str(int(hex_label[ff_i])) )

        #fig.show()
        if len(save_fig):
            plt.savefig(save_fig+'_dim{}.svg'.format(ii))


class subdivided_mesh():
    def __init__(self, VV, EE, FF, level=4):
        """
        # input
        # - VV: numpy array of shape (NVV, 3). three-dimensional coordinates
        # - EE: numpy array of shape (NEE, 2). For each edge (undirected) list the two vertices that define it.
        # - FF: numpy array of shape (NFF, 3). For each face list the three vertices that define it.
        # - level: level of subdivision
        # variables:
        # - VV_sub_np, FF_sub_np, EE_sub_np: list of vertices, faces and edges of subdivided complete mesh
        # - connectivity_list: connectivity_list for complete mesh
        # - FF_sub_to_ee: list of length NFF. for each original face give the corresponding edges of subdivided mesh
        # - boundaries_sub: for each original face give the corresponding boundary-edges in the subdivided mesh
        """
        
        self.VV_np = VV
        self.N_VV = len(VV)
        self.EE_np = EE
        self.N_EE = len(EE)
        self.FF_np = FF
        self.N_FF = len(FF)
        self.level = level
        
        self._boundary_verts = None
        
        print("Subdivide faces of triangle mesh {} times".format(level))
        self.VV_sub_np, self.FF_sub_np, self.EE_sub_np, self.connectivity_list, self.FF_sub_to_ee, self.boundaries_sub \
            = self.subdivide_triangle_mesh()
        
    def boundary_verts(self):
        """
        # output
        # - indices of boundary vertices of orginal mesh
        """
        
        if self._boundary_verts is None:
            
            #self._boundary_verts = get_boundary_verts(self.EE_np, self.FF_np)
            _, _, self._boundary_verts, _ = get_edges(self.VV_np, self.FF_np)

        return  self._boundary_verts
        
    def subdivide_triangle_mesh(self):
        """
        # given a triangular mesh subdivide its faces level times

        # output:
        # - VV2, FF2, EE2: list of vertices, faces and edges of subdivided complete mesh
        # - connectivity: connectivity_list for complete mesh
        # - FF_to_ee: list of length NFF. for each original face give the corresponding edges of subdivided mesh
        # - boundaries: for each original face give the corresponding boundary-edges in the subdivided mesh
        #   (FF_to_ee and boundaries have the same order)
        """
        
        # move to torch because of the subdivision routine
        verts = torch.tensor(self.VV_np, dtype=torch.float32)#, device=device)
        faces = torch.tensor(self.FF_np, dtype=torch.int64) #, device=device)

        boundaries = [list(ff) for ff in self.FF_np]
        # for each coarse triangular face we need a list with edges it contains
        FF_to_ee = [[list(ff[:2]), list(ff[1:]), list(ff[[0,2]])] for ff in self.FF_np]

        flatmesh2 = Meshes(verts=[verts], faces=[faces])
        subdivide = SubdivideMeshes()
        for ii in range(0,self.level):
            flatmesh2 = subdivide(flatmesh2)

            # update information about correspondence to the original coarse triangular faces
            EE2 = flatmesh2.edges_packed().detach().numpy()
            connectivity = connectivity_list(len(flatmesh2.verts_packed()), EE2)
            # update the boundaries of original triangular faces
            for bi, bb in enumerate(boundaries):
                newb = bb.copy()
                for jj in range(len(bb)):
                    c_nn = [vv for vv in connectivity[bb[jj]] if vv in connectivity[bb[(jj+1)%len(bb)]] ]
                    newb = np.append(newb, c_nn)
                newb = sort_connected_vertices(list(newb), EE2)
                boundaries[bi] = newb

            ## go through all edges and find the corresponding new two edges
            for ei, f2e in enumerate(FF_to_ee):
                newf2e = []
                new_v = []
                for jj in f2e:
                    c_nn = [vv for vv in connectivity[jj[0]] if vv in connectivity[jj[1]] ]
                    newf2e += [[c_nn[0],jj[0]]]
                    newf2e += [[c_nn[0],jj[1]]]
                    new_v += [c_nn[0]]
                for ej in EE2:
                    if ej[0] in new_v and ej[1] in new_v:
                        newf2e += [list(ej)]
                FF_to_ee[ei] = list(newf2e)


        # faces, vertices and edges (EE2) of the complete mesh
        FF2 = flatmesh2.faces_packed().detach().numpy()
        VV2 = flatmesh2.verts_packed().detach().numpy()

        return VV2, FF2, EE2, connectivity, FF_to_ee, boundaries
    
    def plot_subdivided_triangles_xy_vert_index(self, figsize=(3,3)):
        """ plot x and y coordinates of each coarse face of the subdivided triangle mesh
        input:
        - figsize: tuple giving size of the figure, defualt: (3,3)
        """
        # treat original triangles separately
        # plot x and y coordinates
        for fei, f2e in enumerate(self.FF_sub_to_ee):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            for ei, ee in enumerate(f2e):
                ax.plot(self.VV_sub_np[ee][:,0], self.VV_sub_np[ee][:,1], c='k',lw=1)
                ax.text(self.VV_sub_np[ee][0][0]+0.02, self.VV_sub_np[ee][0][1]-0.01,  ee[0], alpha=0.5)
                ax.text(self.VV_sub_np[ee][1][0]+0.02, self.VV_sub_np[ee][1][1]-0.01,  ee[1], alpha=0.5)
            bb = self.boundaries_sub[fei]
            for jj in range(len(bb)):
                sel = [bb[jj], bb[(jj+1)%len(bb)]]
                ax.plot(self.VV_sub_np[sel][:,0], self.VV_sub_np[sel][:,1], c='r')

            plt.axis('off')
            plt.tight_layout()    

    def plot_subdivided_mesh_xy_vert_index(self, figsize=(7,7)):
        """ plot x and y coordinates of the subdivided triangle mesh
        input:
        - figsize: tuple giving size of the figure, defualt: (3,3)
        """
        # plot all together
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for ei, ee in enumerate(self.EE_sub_np):
            ax.plot(self.VV_sub_np[ee][:,0], self.VV_sub_np[ee][:,1], c='k',lw=1)
        for bb in self.boundaries_sub:
            for jj in range(len(bb)):
                sel = [bb[jj], bb[(jj+1)%len(bb)]]
                ax.plot(self.VV_sub_np[sel][:,0], self.VV_sub_np[sel][:,1], c='r')

        for vi, vv in enumerate(self.VV_sub_np):
            ax.text(vv[0]+0.02, vv[1]-0.01,  vi)
            
        plt.axis('off')
        plt.tight_layout()

    def create_final_padded_hextrias(self, kernel_size=2, print_progress = False, rotation=0):
        """
        create for every original triangle in FF_np a padded hexaginal triangle array that can be treated by hexagdly
        input:
        - kernel_size: default=2
        - print_progress: boolean, print the progress (default=False)
        - rotation: (0,1 or 2) rotation by x*120 degrees, default=0
        output:
        - list_padded_hexagonal: list of 2D arrays
        """
        
        list_padded_hexagonal = []

        for fei, f2e in enumerate(self.FF_sub_to_ee):
            if print_progress:
                print('Hex-representation and ',kernel_size,'-padding for face ', fei, sep='')

            # select the vertices of the subdivided face
            VV = []
            for vv in f2e:
                if vv[0] not in VV:
                    VV += [vv[0]]
                if vv[1] not in VV:
                    VV += [vv[1]]
                    
            # 1. create the Hex-representation
            corners = list(self.FF_np[fei])
            if rotation in [1,2]:
                corners = corners[rotation:] + corners[:rotation]
                
            flat_hexagonal = self.match_the_triangle_to_hextria(VV, f2e, 
                                                    boundary = self.boundaries_sub[fei], corners = corners) 

            #still without the padding
            #plot_hextensor( torch.tensor(flat_hexagonal)[[0]], figname='tensor',figsize=(5,5))

            # 1. padding kernelsize 2
            tensor = self.padding_for_hextria_triangle(flat_hexagonal, VV, kernel_size=kernel_size)

            #plot_hextensor( torch.tensor(tensor)[[0]], figname='tensor',figsize=(5,5))

            list_padded_hexagonal += [tensor]
        return list_padded_hexagonal
    


    
    def match_the_triangle_to_hextria(self, VV, EE, boundary, corners=[0,1,2]):
        """
        # given a triangular mesh with regular vertices, a flat hexagonal triangle is created that has the indices of the corresponding vertices. The hexagonal triangle can be treated by the hexagdly-package
        # input:
        # - VV (array of int): Vertices of the triangle
        # - EE: list of tuples that contain the edges
        # - boundary: sorted indices of boundary-vertices of the triangle OR NONE (only works if VV = range(len(VV)))
        # - connectivity: list of lenfth N_VV. Indicates for every vertex its direct neighbors OR NONE (only works if VV = range(len(VV)))
        # - corners: list of length 3 indicating which vertices are the corners (corners[0]: lower left, corners[1]: upper left, corners[2]: right)
        # output:
        # - flat_hexagonal: array of size (1,1, height, width), padding is indicated by -1, other elements indicate the corresponding vertex in range(N_VV)
        """
        
        height = int(2**self.level) + 1
        width  = int(2**self.level) + 1
        
        N_VV = len(VV)

        flat_hexagonal = np.zeros((1,1, height, width)) -1

        # 1. do the padding
        for cc in range(width): # column
            flat_hexagonal[0,0,:int(cc/2),cc] = -20
            if int((cc+1)/2) != 0:
                flat_hexagonal[0,0,-int((cc+1)/2):,cc] = -20

        # 2. fit the corners        
        flat_hexagonal[0,0,0,0] = corners[1]
        flat_hexagonal[0,0,-1,0] = corners[0]
        flat_hexagonal[0,0,int(height/2),-1] = corners[2]

        # 3. get connectivity
        # is given

        # 4. get boundary
        # is given

        # 5. fit the corners[0]-corners[1] boundary
        # reorder the boundary if necessary
        # left upper corner first (corner [1]), left lower corner second (corner [0]), right corner last (corner[2])
        if boundary.index(corners[1]) != 0:
            boundary = boundary[boundary.index(corners[1]):] + (boundary[:boundary.index(corners[1])])
        if boundary.index(corners[2]) < boundary.index(corners[0]):
            boundary = boundary[::-1]
            boundary = [boundary[-1]] + boundary[:-1]

        # match the left side
        flat_hexagonal[0,0,:,0] = boundary[0:boundary.index(corners[0])+1]

        # 6. fit the other columns
        # match column by column
        # list of already positioned vertices
        matched_vv = boundary[0:boundary.index(corners[0])+1]
        for cc in range(1,height-1):
            for rr in range(int(cc/2), height-int((cc+1)/2)):

                # find neighbors in column to left
                comp_cc = int(cc-1)
                if (cc%2)==1:
                    neighbors = [int(flat_hexagonal[0,0,rr,cc-1]), int(flat_hexagonal[0,0,rr+1,cc-1])]
                else:
                    neighbors = [int(flat_hexagonal[0,0,rr,cc-1]), int(flat_hexagonal[0,0,rr-1,cc-1])]

                # common neighbor of neighbors that is not positioned yet
                common_nn = [ii for ii in self.connectivity_list[neighbors[0]] if ii in self.connectivity_list[neighbors[1]] and ii not in matched_vv]

                # if first column after boundary we have to make sure to the select the neighbors to correct side
                if cc == 1:
                    for ij in range(len(common_nn)):
                        if common_nn[ij] in VV:
                            flat_hexagonal[0,0,rr,cc] = common_nn[ij]
                            # add to list of already positioned vertices
                            matched_vv += [common_nn[ij]]
                            break
                    
                # if not first column after boundary
                else:
                    flat_hexagonal[0,0,rr,cc] = common_nn[0]
                    matched_vv += [common_nn[0]]

        return flat_hexagonal
    
    
    def padding_for_hextria_triangle(self, tensor, VV, kernel_size=2):
        """
        # given a coarse face of the triangular mesh that is already meshed to hex, do the padding for given kernel_size if there are neighboring faces
        # input:
        # - tensor (array of size (1,1, height, width)): tringular mesh without padding, empty vertices have value -20
        # - VV (array of int): Vertices (at the moment only for regular vertices)
        # - kernel_size (int)
        # output:
        # - flat_hexagonal: array of size (1,1, height+padding, width+padding)
        """
        
        height, width = tensor.shape [2:]

        # 1. add columns and rows for the padding
        for ii in range(1,kernel_size+1):
            #print('Add one column/row at each side')
            new_tensor = np.ones((1,1,height+(ii)*2, width+(ii)*2))*(-20)
            new_tensor[:,:,1:-1,1:-1] = tensor
            tensor=new_tensor.copy()    

        # 1.b if uneven add column to left and right
        if kernel_size%2 == 1:
            new_tensor = np.ones((1,1,height+(kernel_size)*2, width+(kernel_size)*2+1+1))*(-20)
            new_tensor[:,:,:,1:-1] = tensor
            tensor=new_tensor.copy()

        matched_VV = VV.copy()

        # 2. add neighbors round by round to have a correct padding
        for jj in range(kernel_size):
            # one round for each stride

            # a) the the kernel size is odd, an extra column has been added
            extra_column = kernel_size%2

            # b) find the vertices in the hexagonal grid, that have to be filled up. they are saved in the list rounding
            # rounding: list of index tupels (row,column) in the 2D-array
            # left 
            rounding = [[ ii, 
                         kernel_size+extra_column-1 - jj] for ii in range( kernel_size-1-int(jj/2), height+(kernel_size)+int((jj+1)/2))]
            
            # change position of the first one since it has only one neighbor
            tmp = rounding[0].copy()
            rounding[0] = rounding[1].copy()
            rounding[1] = tmp

            # left upper corner
            # right lower corner
            if jj>=1:        
                # take values from ii+1 iterations before
                rounding += [[  kernel_size-1-int((jj-ii-1)/2) - ii-1   , kernel_size+extra_column - jj+ii] for ii in range(jj)]
                rounding += [[  height+(kernel_size)+int((jj-ii)/2) + ii   , kernel_size+extra_column - jj+ii] for ii in range(jj)]

            # top
            rounding += [[ int((ii+extra_column)/2)-jj + int(kernel_size/2)-1, 
                          ii +extra_column] for ii in range(kernel_size, width+(kernel_size)+1+jj )]
            #bottom
            rounding += [[ height+2*(kernel_size)-1 - (int((ii+extra_column+1)/2)-jj + int(kernel_size/2)-1), 
                          ii +extra_column] for ii in range(kernel_size, width+(kernel_size)+1+jj)]

            #right
            if jj>=1:
                f_r = int((width+(kernel_size)+jj+extra_column)/2)-jj + int(kernel_size/2)-1
                l_r = height+2*(kernel_size)-1 - (int((width+(kernel_size)+jj+extra_column+1)/2)-jj + int(kernel_size/2)-1)
                rounding += [[ ii, 
                              width+(kernel_size)+jj + extra_column] for ii in range(f_r+1,l_r)]


            # if rounding should be visualized
            #for rr in rounding:
            #    loss_function_weight[0,0][rr[0],rr[1]] = 2
            #plot_hextensor( torch.tensor(loss_function_weight)[[0]], figname='tensor',figsize=(7,7))


            ## c) to all the indexes in list rounding find the corresponding vertex in the mesh 

            # register vertices that have not 6 neighbors
            # TODO: for this we need to have a list of the overall boundary vertices
            #notregular = [vv for vv in VV if len(connectivity[vv])!=6 ]
            #print('Not 6 neighbors:',notregular)
            
            again = []

            for rr in rounding:
                #print('match',rr)
                # find neighbors
                nnei = get_neigbors_in_hex(*rr,height+(kernel_size)*2-1, width+(kernel_size)*2+2*extra_column-1)
                # find neighbors that have a value
                nnei_with_value = [tensor[0,0,xx[0],xx[1]] for xx in nnei if tensor[0,0,xx[0],xx[1]] >=0]
                # if less than 2: impossible!
                if len(nnei_with_value)>1:
                    # find neighbors, that are also neighbors -> than we have a face
                    for xn, xx in enumerate(nnei_with_value):
                        for yy in nnei_with_value[xn+1:]:
                            if yy in self.connectivity_list[int(xx)]:
                                common_nn = [vv for vv in self.connectivity_list[int(xx)] if vv in self.connectivity_list[int(yy)]]            

                    # from those select the one, that has not been selected yet
                    matched = [cc for cc in common_nn if cc not in matched_VV]      
                    if len(matched) != 0:
                        tensor[0,0,rr[0],rr[1]] = matched[0]
                        matched_VV += [matched[0]]
                        #print('matched:', matched)
                    else:
                        tensor[0,0,rr[0],rr[1]] = -(jj+1)
                    #    print('match',rr)
                    #    print('nothing matched. maybe boundary?')
                elif len(nnei_with_value)>0:
                    # only one enighbor known. test again once the rest is matched.
                    again += [rr]
                elif len(nnei_with_value)==0:
                    tensor[0,0,rr[0],rr[1]] = -(jj+1)
                    
            for rr in again:
                #print('match',rr)
                # find neighbors
                nnei = get_neigbors_in_hex(*rr,height+(kernel_size)*2-1, width+(kernel_size)*2+2*extra_column-1)
                # find neighbors that have a value
                nnei_with_value = [tensor[0,0,xx[0],xx[1]] for xx in nnei if tensor[0,0,xx[0],xx[1]] >=0]
                # if less than 2: impossible!
                if len(nnei_with_value)>1:
                    # find neighbors, that are also neighbors -> than we have a face
                    for xn, xx in enumerate(nnei_with_value):
                        for yy in nnei_with_value[xn+1:]:
                            if yy in self.connectivity_list[int(xx)]:
                                common_nn = [vv for vv in self.connectivity_list[int(xx)] if vv in self.connectivity_list[int(yy)]]            

                    # from those select the one, that has not been selected yet
                    matched = [cc for cc in common_nn if cc not in matched_VV]      
                    if len(matched) != 0:
                        tensor[0,0,rr[0],rr[1]] = matched[0]
                        matched_VV += [matched[0]]
                    else:
                        tensor[0,0,rr[0],rr[1]] = -(jj+1)
                else:
                    tensor[0,0,rr[0],rr[1]] = -(jj+1)
                        
        return tensor