#!/usr/bin/env python3

import sys
from typing import Tuple, Union
import torch


def sample_points_from_boundary(
    meshes, boundary_edges, num_samples: int = 10000
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert a mesh to a pointcloud by uniformly sampling points on
    the boundary of the mesh with probability proportional to the edge length.

    Args:
        meshes: A Meshes object with a batch of N meshes. Only works for one mesh
        boundary_edges: list of indices of the boundary vertices
        num_samples: Integer giving the number of point samples per mesh.

    Returns:
        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.

    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    edges = meshes.edges_packed()
    
    ############## filter the edges
    edges = edges[boundary_edges]            
    
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    
    if num_meshes==1:
    
        # Intialize samples tensor with fill value 0 for empty meshes.
        samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

        # Only compute samples for non empty meshes
        with torch.no_grad():
            #areas, _ = _C.face_areas_normals(
            lengths = (verts[edges[:,0]] - verts[edges[:,1]]).pow(2).sum(1)
            # (len(boundary_edges))
            #max_faces = meshes.num_faces_per_mesh().max().item()
            #areas_padded = packed_to_padded(
            #    areas, mesh_to_face[meshes.valid], max_faces
            #)  # (N, F)

            # TODO (gkioxari) Confirm multinomial bug is not present with real data.
            sample_face_idxs = lengths.multinomial(
                num_samples, replacement=True
            )  # (num_samples)
            #sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

        # Get the vertex indices of the sampled edges.
        ee0, ee1 = edges[sample_face_idxs][:, 0], edges[sample_face_idxs][: , 1]
        
        # Randomly generate coords for linear combination.
        w0 = torch.rand(num_samples,1, device=meshes.device).repeat(1,3)
        v0 = verts[ee0]  # (N, num_samples, 3)
        v1 = verts[ee1]
        # take weighted average
        samples[meshes.valid] = (
            w0 * v0 + (1-w0) * v1
        )

    return samples