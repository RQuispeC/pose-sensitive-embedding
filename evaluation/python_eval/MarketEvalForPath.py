import csv
import numpy as np
from scipy import spatial
from evaluation.python_eval.evaluation import evaluation

def readCSV(path):
    output = []
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            output.append(row)
    return np.array(output)

def evalMarketWithPath(evalPath):
    querymat=readCSV(evalPath + '/query/features.csv')
    queryLab=readCSV(evalPath + '/query/labels.csv')
    queryCam=readCSV(evalPath + '/query/cameras.csv')

    testmat=readCSV(evalPath + '/test/features.csv')
    testLab=readCSV(evalPath + '/test/labels.csv')
    testCam=readCSV(evalPath + '/test/cameras.csv')

    print(evalPath)

    noRerankingDist = spatial.distance.cdist(testmat, querymat, 'cosine')
    rec_rates, mAP, _, _ = evaluation(noRerankingDist, testLab, queryLab, testCam, queryCam)

    result.rec_rates = 100 * rec_rates
    result.mAP = 100 * mAP
    return result
