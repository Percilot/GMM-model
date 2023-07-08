import os
import sys
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

if __name__ == "__main__":
    files=[os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1])]
    all_mfcc_of_a_user = np.asarray(())

    for f in files:
        known_mfcc = np.loadtxt(f, delimiter=',')
        if all_mfcc_of_a_user.size == 0:
            all_mfcc_of_a_user = known_mfcc
        else:
            all_mfcc_of_a_user = np.vstack((all_mfcc_of_a_user, known_mfcc))
    
    gmm_model = GaussianMixture(n_components=8, covariance_type='diag', max_iter=200, n_init=3)
    gmm_model.fit(all_mfcc_of_a_user)

    pickleFile = sys.argv[2]
    outputPath = sys.argv[3]

    pickle.dump(gmm_model, open(outputPath + '/' + pickleFile, 'wb'))

