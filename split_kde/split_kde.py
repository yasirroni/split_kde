import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

def split_kde(x, model=None, start_end=None, n_groups=None, lower_bound=None, upper_bound=None, max_iter=100):
    '''
    x: data
    model: sklearn.neighbors.KernelDensity
    start_end: (optional) linspace start and end point
    n_groups: (optional) specify number of groups after splitted
    lower_bound: (optional) if n_groups, lower bound for binary search
    upper_bound: (optional) if n_groups, upper bound for binary search
    max_iter: maximum number of iteration
    '''
    if start_end == None:
        start = min(x)
        end = max(x)
    else:
        start, end = (start_end)
    
    if model == None:
        model = KernelDensity(kernel='gaussian', bandwidth=(end-start)/100)
    
    model.s_ = np.linspace(start,end)
    model.fit(x)
    model.e_ = model.score_samples(model.s_.reshape(-1,1))
    mi = argrelextrema(model.e_, np.less)[0]
    model.mi_ = mi
    if n_groups:
        # binary_search
        lower_bound = 0
        iteration = 0
        while len(mi) != (n_groups-1):
            if len(mi) > (n_groups-1):
                # increase kernel
                lower_bound = model.bandwidth
                if upper_bound:
                    model.bandwidth = lower_bound + (upper_bound-lower_bound) / 2
                else:
                    model.bandwidth = lower_bound * 2
            else:
                # decrease kernel
                upper_bound = model.bandwidth
                model.bandwidth = lower_bound + (upper_bound-lower_bound) / 2
            model.fit(x)
            model.e_ = model.score_samples(model.s_.reshape(-1,1))
            mi = argrelextrema(model.e_, np.less)[0]
            iteration += 1
            if iteration == max_iter:
                raise Exception('No convergence. Try remove start_end or reduce the difference, or increase max_iter')
    
    model.split_points_ = model.s_[mi]    
    if model.split_points_.size == 0:
        model.splitted_ = [np.array(x).flatten()]
        model.labels_ = [0]*len(x)
        return model
    else:
        model.splitted_ = get_splitted_(x, model.split_points_)
    model.labels_ = get_labels_(x, model.split_points_)
    return model

def get_split_points_(e, s):
    mi = get_mi(e)  
    return s[mi]

def get_splitted_(x, split_points_):
    splitted_ = [x[x <= split_points_[0]]]
    for idx in range(len(split_points_))[:-1]:
        splitted_.append(x[(x > split_points_[idx]) * (x <= split_points_[idx+1])])  
    splitted_.append(x[x > split_points_[-1]])
    return splitted_

def get_labels_(x, split_points_):
    labels_ = []
    for val in x:
        start_idx = 0
        label_ = get_label_(val, start_idx, split_points_)
        labels_.append(label_)
    return np.array(labels_)

def get_label_(val, idx, split_points_):
    try:
        if val < split_points_[idx]:
            return idx
        else:
            idx = get_label_(val, idx+1, split_points_)
            return idx
    except:
        return len(split_points_)
