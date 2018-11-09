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

tt = 10 # TAU
aa = 0.5 # ALPHA

if __name__ == "__main__":
        
    for nDIM in range(1, 11):
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

        # load weights
        weights = sio.loadmat('../../../results/' + datatask + '/wts_ESN_Xy_' + str(tt) + '_' + str(aa) + '.mat')
        wts = weights['weights']
        # sort_wts = np.sort(wts, axis = 0) # low to high
        sort_ord = np.argsort(wts, axis = 0)
        
        top = sort_ord[-nDIM:] # most weighted is the bottom most
        bot = sort_ord[:nDIM] # most weighted is top most
        # concatenate
        dim = np.concatenate((top, bot), axis = 0)
        dim = dim.tolist()
        
        X = np.squeeze(X[:, dim])
    
        # cross val    
        train_acc = []
        val_acc = []
        kf = KFold(n_splits = kFOLD)
        for train, val in kf.split(list(range(0, kSUB))):
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
            
            #ith point acc
            v_acc = []
            y_hat = lr.predict(X_val)
            for ii in range(0, BLOCK_SIZE):
                y_val_i = y_val[ii::BLOCK_SIZE]
                y_hat_i = y_hat[ii::BLOCK_SIZE]
                indicator_correct = [1 for a, b in zip(y_val_i, y_hat_i) if a==b]
                acc = sum(indicator_correct) / len(y_val_i)
                v_acc.append(acc)
            t_acc = []
            y_hat = lr.predict(X_train)
            for ii in range(0, BLOCK_SIZE):
                y_train_i = y_train[ii::BLOCK_SIZE]
                y_hat_i = y_hat[ii::BLOCK_SIZE]
                indicator_correct = [1 for a, b in zip(y_train_i, y_hat_i) if a==b]
                acc = sum(indicator_correct) / len(y_train_i)
                t_acc.append(acc)
      
            # pointwise acc    
            train_acc.append(t_acc)
            val_acc.append(v_acc)
            
            
        print("Mean pointwise train accuracy is " + str(np.mean(train_acc)))
        print("Mean pointwise val accuracy is " + str(np.mean(val_acc)))
        
        # save result matrices for plots
        results = {}
        results['train_acc'] = train_acc
        results['val_acc'] = val_acc
        sio.savemat('../../../results/' + datatask + '/crossval_dim_ESN_Xy_ith_' + str(tt) + '_' + str(aa) + '_' + str(nDIM) + '.mat', results)