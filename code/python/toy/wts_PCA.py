"""
train-val split: subjectwise
cross val
data: PCA data
save wts to results
"""

import scipy.io as sio
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

datatask = 'toy'
kSEED = 42
kFOLD = 5
kSUB = 10

TAU = [1, 2, 5, 10]
ALPHA = [0.2, 0.5, 0.8]

if __name__ == "__main__":
        
    for tt in TAU:
        for aa in ALPHA:
            # load data
            data = sio.loadmat('../../../data/' + datatask + '/ESN/data_PCA_ESN_Xy_' + str(tt) + '_' + str(aa) + '.mat')
            data = data['pc_ESN']
            data = data[0,0]
            X = data['X']
            X = minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
            y = data['y']
            latent = data['latent']
            y = y.flatten()
            BLOCK_SIZE = int(data['BLOCK_SIZE'])
            SUB_ID = data['SUB_ID']
            SUB_ID = list(SUB_ID)

            # save weights from a single fold   
            train_acc = []
            val_acc = []
            kf = KFold(n_splits = kFOLD)
            t = kf.split(list(range(0, kSUB)))
            t = list(t)
            train, val = t[0]
        
            # locations based on sub IDs, sub IDs range from 1 to kSUB
            train_loc = []
            for ii in train:
                train_loc.extend([i for i, j in enumerate(SUB_ID) if j == ii + 1])
            val_loc = []
            for jj in val:
                val_loc.extend([i for i, j in enumerate(SUB_ID) if j == jj + 1])
            
            # lr
            X_train, X_val, y_train, y_val = X[train_loc], X[val_loc], y[train_loc], y[val_loc]
            lr = LogisticRegression(penalty='l2', random_state = kSEED + 1)
            lr.fit(X_train, y_train)
            
            weights = lr.coef_.T
            
            # pointwise acc    
            train_acc.append(lr.score(X_train, y_train))
            val_acc.append(lr.score(X_val, y_val))      
                
            # print("Mean pointwise train accuracy is " + str(np.mean(train_acc)))
            # print("Mean pointwise val accuracy is " + str(np.mean(val_acc)))
            
            # save result matrices for plots
            results = {}
            results['train_acc'] = train_acc
            results['val_acc'] = val_acc
            results['weights'] = weights
            results['latent'] = latent
            sio.savemat('../../../results/' + datatask + '/wts_ESN_Xy_' + str(tt) + '_' + str(aa) + '.mat', results)