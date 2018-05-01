#This function is from https://github.com/zhunzhong07/person-re-ranking by Zhong et al.

import numpy as np
import math
from evaluation.python_eval.compute_AP import compute_AP

def evaluation(dist, label_gallery, label_query, cam_gallery, cam_query):
    junk0 = np.array([i for i, j in enumerate(label_gallery) if j == -1])
    ap = np.zeros((dist.shape[1]))
    CMC = np.zeros(dist.shape)
    r1_pairwise = np.zeros((dist.shape[1], 6));#pairwise rank 1 precision  
    ap_pairwise = np.zeros((dist.shape[1], 6));#pairwise average precision

    for k in range(dist.shape[1]):
        score = dist[:, k]
        q_label = label_query[k]
        q_cam = cam_query[k]
        pos = np.array([i for i, j in enumerate(label_gallery) if j == q_label])
        pos2 = np.array([i for i, x in enumerate(cam_gallery[pos]) if x!=q_cam])
        if len(pos2) > 0:
            good_image = pos[pos2]
        else:
            good_image = np.array([])
        pos3 = np.array([i for i, x in enumerate(cam_gallery[pos]) if x==q_cam])
        if len(pos3) > 0:
            junk = pos[pos3]
        else:
            junk = np.array([])
        junk_image = np.append(junk0, junk)
        index = np.argsort(score)
        ap[k], CMC[:, k] = compute_AP(good_image, junk_image, index)
        #ap_pairwise[k, :] = compute_AP_multiCam(good_image, junk, index, q_cam, cam_gallery); # compute pairwise AP for single query
        #r1_pairwise[k, :] = compute_r1_multiCam(good_image, junk, index, q_cam, cam_gallery); # pairwise rank 1 precision with single query
    CMC = np.sum(CMC, axis = 1).astype(np.double)/dist.shape[1]
    CMC = np.transpose(CMC)
    isNan = np.array([i for i, j in enumerate(ap) if math.isnan(j)])
    if len(isNan) > 0:
        ap[isNan] = 0
    Map = np.sum(ap)/len(ap)

    return CMC, Map, r1_pairwise, ap_pairwise