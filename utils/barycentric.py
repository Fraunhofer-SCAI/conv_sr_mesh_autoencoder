import numpy as np

def Barycentric(p, a, b, c):
    #// Compute barycentric coordinates (u, v, w) for
    #// point p with respect to triangle (a, b, c)
    v0 = b - a 
    v1 = c - a 
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        print('denom is zero')
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return u, v, w
    

def move_projection_inside_triangle(bary_coords):
    # bary_coords: numpy array of size (3) (only for 1d)
    # if two coordinates are negative, project to corner with positive factor
    # if one coordinate is negative, project to line between corners with positive factor by rescaling the factors to sum one (this is not an orthogonal projection)
    if bary_coords[0] < 0:
        new_0 = 0
        new_1 = bary_coords[1] / (bary_coords[1]+bary_coords[2]) 
        new_2 = bary_coords[2] / (bary_coords[1]+bary_coords[2]) 
        if bary_coords[1] < 0:
            # project to corner
            new_1 = 0
            new_2 = 1
        elif bary_coords[2] < 0:
            # project to corner
            new_1 = 1
            new_2 = 0
    elif bary_coords[1] < 0:
        new_0 = bary_coords[0] / (bary_coords[0]+bary_coords[2]) 
        new_1 = 0
        new_2 = bary_coords[2] / (bary_coords[0]+bary_coords[2])         
        if bary_coords[2] < 0:
            # project to corner
            new_0 = 1
            new_2 = 0
    elif bary_coords[2]<0:
        new_0 = bary_coords[0] / (bary_coords[0]+bary_coords[1]) 
        new_1 = bary_coords[1] / (bary_coords[0]+bary_coords[1])  
        new_2 = 0
    else:
        return bary_coords
    return np.array([new_0, new_1, new_2])