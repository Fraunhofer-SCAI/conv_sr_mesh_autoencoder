import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80


def clean_mesh_get_edges(VV, FF):
    # given list of Vertices and Faces, clean face duplicates, get list of edges and boundary/non-manifold information 
    # input:
    # - VV: list of vertices
    # - FF: list of triplets defining faces
    # output:
    # - VV: filterted vertices (delete vertices without face)
    # - FF: faces
    # - EE: edges
    # - boundary_edges, boundary_vertices, nonmanifold_edges
    
    FF_tmp = np.sort(FF, axis=1)
    FF_tmp, unique_faces_locs = np.unique(FF_tmp, axis=0, return_index=True)
    
    # filter vertices: this is dangerous. dont do it
    
    # edges
    edge_1=FF_tmp[:,0:2]
    edge_2=FF_tmp[:,1:]
    edge_3=np.concatenate([FF_tmp[:,:1], FF_tmp[:,-1:]], axis=1)
    EE=np.concatenate([edge_1, edge_2, edge_3], axis=0)
    
    # delete duplicates
    unique_edges_trans, unique_edges_locs, edges_counts=np.unique(EE[:,0]*(10**5)+EE[:,1], return_index=True, return_counts=True)
    EE=EE[unique_edges_locs,:]
    
    boundary_edges = np.where(edges_counts==1)[0]
    boundary_vertices = np.unique( EE[boundary_edges])
    nonmanifold_edges = np.where(edges_counts>2)[0]
    
    return VV, FF[unique_faces_locs], EE, boundary_edges, boundary_vertices, nonmanifold_edges

def get_edges(VV, FF):
    # given list of Vertices and Faces, get list of edges and boundary/non-manifold information 
    # input:
    # - VV: list of vertices
    # - FF: list of triplets defining faces
    # output:
    # - EE: edges
    # - boundary_edges, boundary_vertices, nonmanifold_edges
    
    FF_tmp = np.sort(FF, axis=1)
    #FF_tmp, unique_faces_locs = np.unique(FF_tmp, axis=0, return_index=True)
    
    # edges
    edge_1=FF_tmp[:,0:2]
    edge_2=FF_tmp[:,1:]
    edge_3=np.concatenate([FF_tmp[:,:1], FF_tmp[:,-1:]], axis=1)
    EE=np.concatenate([edge_1, edge_2, edge_3], axis=0)
    
    # delete duplicates
    unique_edges_trans, unique_edges_locs, edges_counts=np.unique(EE[:,0]*(10**5)+EE[:,1], return_index=True, return_counts=True)
    EE=EE[unique_edges_locs,:]
    
    boundary_edges = np.where(edges_counts==1)[0]
    boundary_vertices = np.unique( EE[boundary_edges])
    nonmanifold_edges = np.where(edges_counts>2)[0]
    
    return EE, boundary_edges, boundary_vertices, nonmanifold_edges


def connectivity_list(N_VV, EE):
    # given Vertices from 0 to N_VV-1 and undirected edges, define the connectivity list
    # input:
    # - N_VV: number of vertices
    # - EE: list of tuples that define the connecting undirected edges
    # output:
    # - connectivity: list of length N_VV that contains for every vertex a list of its direct neighbors
    connectivity = []
    for vv in range(N_VV):
        con_nn = []
        for ee in EE:
            if vv == ee[0] and ee[1] not in con_nn:
                con_nn += [ee[1]]
            if vv == ee[1] and ee[0] not in con_nn:
                con_nn += [ee[0]]
        connectivity += [con_nn]
    return connectivity


def sort_connected_vertices(list_vv, edges):
    # input
    # - list_vv: list of the vertex indices that form a circle
    # - edges: list of tuples, that indicate the edges between two vertices
    # output:
    # - list_vv_sorted: sorted list_vv 
    list_vv_sorted = [list_vv[0]]
    list_vv.remove(list_vv[0])
    nn = len(list_vv)
    for jj in range(nn):
        for ee in edges:
            if list_vv_sorted[-1] == ee[0]:
                if ee[1] in list_vv:
                    list_vv_sorted += [ee[1]]
                    list_vv.remove(ee[1])
                    break
            if list_vv_sorted[-1] == ee[1]:
                if ee[0] in list_vv:
                    list_vv_sorted += [ee[0]]
                    list_vv.remove(ee[0])
                    break
    return list_vv_sorted


def get_neigbors_in_hex(row, col, max_row, max_col, min_row=0, min_col=0):
    # for a tensor representing a hexagonal mesh, where every vertex has six neighbors, output the indices of the neighboring vertices
    # input:
    # - row, col: index of the vertex whose neighbors we are looking for
    # - max_row, max_col: highest possible index for row/column
    # - min_row, min_col: lowest possible index for row/column
    # output:
    # - mask (list of int-tuples): index up to 6 neighbors
    
    # straigth up and down are always neighbors
    neighbors = [[row+1, col], [ row-1, col], 
                 [row, col+1], [row, col-1]]
    # if column is odd, the lower row contains the neighbors
    if col%2 == 0:
        neighbors += [[row-1, col-1], [row-1, col+1]]
    # if column is even, the upper row contains the neighbors
    else:
        neighbors += [[row+1, col-1], [row+1, col+1]]
        
    # check the possible lowest and highest indices
    mask = [nn for ii, nn in enumerate(neighbors) if (nn[0]<=max_row and nn[0]>=min_row and nn[1]<=max_col and nn[1]>=min_col) ]
        
    return mask


def plot_mesh_over_time(vertices, edges, tt, second_vertices = None, second_edges = None, title='', save_plot = '', save_gif=False):
    # plot mesh
    # input:
    # - vertices: shape (#timesteps, #vertices, 3)
    # - edges: list with edges
    # - tt: list of timesteps to plot
    # - second_vertices: funtion plots a second mesh if given
    # - second_edges: funtion plots a second mesh if given
    # - title: title of the figure
    # - save_plot: functions saves the plot to given directory under name 'mesh_projected_t_{}.png'
    # - save_gif: bool (save plots as a gif yes or no)
    
    times = np.shape(vertices)[0]
    
    if times < max(tt):
        tt = np.arange(0,times,10)
        
    #print(range(0,np.min([times,100]),10))

    for t0 in tt:
    
        fi4 = plt.figure(figsize=[12.8, 9.6])
        plt.title(title)
        plt.axis('off')

        gs = gridspec.GridSpec(1, 1) 

        plot41 = fi4.add_subplot(gs[0], projection='3d')  
        #plot42 = fi4.add_subplot(gs[1])

        plot41.set_aspect('equal')
        plot41.axis('equal')
        plot41.axis('off') 

        if second_vertices is not None and second_edges is not None:
            for ee in second_edges:
                plot41.plot(second_vertices[t0,ee,0], second_vertices[t0,ee,2], -second_vertices[t0,ee,1], c='k', alpha = 0.3, lw=0.5)
        
        for ee in edges:
            plot41.plot(vertices[t0,ee,0], vertices[t0,ee,2], -vertices[t0,ee,1], c='darkred', alpha = 0.6)
            #plot41.legend()

        plot41.view_init(180, 105+190)
        
        plt.tight_layout()

        plt.savefig(save_plot+'mesh_projected_t_{}.png'.format(str(tt[t0]).zfill(3)))
        fi4.show()
        plt.close(fi4)
        
    if save_gif:
        # Save Video in the same Folder
        gif_name = save_plot + 'mesh_projected_all_t.gif'
        # get a list of all the png-files in the directory
        images = [img for img in os.listdir(save_plot) if img.endswith(".png") and 'mesh_projected_t' in img]
        gif = []
        # add the frames
        for image in sorted(images):
            gif.append(imageio.imread(os.path.join(save_plot, image)))

        imageio.mimsave(gif_name, gif, duration=0.2)