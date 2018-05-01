#This function is from https://github.com/zhunzhong07/person-re-ranking by Zhong et al.
import numpy as np

def compute_AP(good_image, junk_image, index):
    cmc = np.zeros((len(index)))
    ngood = len(good_image)

    old_recall = 0
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(len(index)):
        flag = 0
        if len([i for i, j in enumerate(good_image) if j == index[n]]) != 0:
            cmc[n - njunk:] = 1
            flag = 1 # good image
            good_now = good_now + 1
        if len([i for i, j in enumerate(junk_image) if j == index[n]]) != 0:
            njunk = njunk + 1;
            continue # junk image 
        if flag == 1: #good
            intersect_size = intersect_size + 1

        recall = intersect_size / ngood
        precision = intersect_size/(j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision)/2)
        old_recall = recall
        old_precision = precision
        j = j + 1 
        
        if good_now == ngood:
            break
    return ap, cmc
