import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

def split_kde(x, model=None, start_end=None):
    if model == None:
        model = KernelDensity(kernel='gaussian', bandwidth=(max(x)-min(x))/100)
    model.fit(x)
    if start_end == None:
        start = max(x)
        end = min(x)
    else:
        start, end = (start_end)
    model.s_ = np.linspace(start,end)
    model.e_ = model.score_samples(model.s_.reshape(-1,1))
    model.split_points_ = get_split_points_(model.e_, model.s_)
    if model.split_points_.size == 0:
        model.splitted_ = [x]
        model.labels_ = [0]*len(x)
        return model
    else:
        model.splitted_ = get_splitted_(x, model.split_points_)
    model.labels_ = get_labels_(x, model.split_points_)
    return model

def get_split_points_(e, s):
    mi = argrelextrema(e, np.less)[0]
    return s[mi]

def get_splitted_(x, split_points_):
    splitted_ = [x[x < split_points_[0]]]
    for idx in range(len(split_points_))[:-1]:
        splitted_.append(x[(x >= split_points_[idx]) * (x < split_points_[idx+1])])  
    splitted_.append(x[x >= split_points_[-1]])
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
