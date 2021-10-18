import numpy as np

def normalize(data, ntype='mean-0-std-1',return_minmax=False):
    # data of shape (timesteps, vertices, number of features)
    if ntype=='range-0,1':
        # data between 0 and std 1 for each mesh at different timesteps
        vval_min = np.min(data)
        vval_max = np.max(data)
        return (data - vval_min) / (vval_max - vval_min) 
    if ntype=='range-0,1-mean-0':
        vval_mean = np.repeat(np.reshape(np.mean(data,axis=(1)), (-1,1,3)), data.shape[1], axis=1)
        data = (data - vval_mean)
        vval_min = np.min(data)
        vval_max = np.max(data)
        if return_minmax:
            return (data - vval_min) / (vval_max - vval_min), vval_min, vval_max
        else:
            return (data - vval_min) / (vval_max - vval_min) 
    elif ntype=='mean-0-std-1':
        # mean 0 and std 1 for each mesh at different timesteps and features
        vval_mean = np.repeat(np.reshape(np.mean(data,axis=(1)), (-1,1,3)), data.shape[1], axis=1)
        vval_std = np.repeat(np.reshape(np.std(data-vval_mean,axis=(1)), (-1,1,3)), data.shape[1], axis=1)
        return (data - vval_mean) / vval_std
    else:
        return data