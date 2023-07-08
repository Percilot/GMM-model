import sys
import os
import pickle
import numpy as np

if __name__ == "__main__":
    modelPath = sys.argv[1]
    sourcePath = sys.argv[2]
    realLabel = sys.argv[3]
    gmm_files = [os.path.join(modelPath, fname) for fname in os.listdir(modelPath) if fname.endswith('.gmm')]
    models = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    
    labels = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
    print(labels)

    test_files = [os.path.join(sourcePath, f) for f in os.listdir(sourcePath)]
    j = 0
    for f in test_files:
        Unknown_mfcc = np.loadtxt(f, delimiter=',')
        scores = None
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            ## 求概率
            scores = np.array(gmm.score(Unknown_mfcc).reshape(1,-1))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        if (realLabel == labels[winner]):
            j = j + 1
    print("j: ", j, " len: ", len(test_files))
    print("识别率是: ", j / len(test_files))
 
