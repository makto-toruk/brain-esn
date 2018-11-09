"""
train-val split: subjectwise
cross val
ith point accuracy
"""

import scipy.io as sio
import numpy as np
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
            data = sio.loadmat('../../../data/' + datatask + '/ESN/data_ESN_Xy_' + str(tt) + '_' + str(aa) + '.mat')
            data = data['data_ESN']
            data = data[0,0]
            X = data['X']
            y = data['y']
            y = y.flatten()
            BLOCK_SIZE = int(data['BLOCK_SIZE'])
            SUB_ID = data['SUB_ID']
            SUB_ID = list(SUB_ID)

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
                
                
            #print("Mean pointwise train accuracy is " + str(np.mean(train_acc)))
            #print("Mean pointwise val accuracy is " + str(np.mean(val_acc)))
            
            # save result matrices for plots
            results = {}
            results['train_acc'] = train_acc
            results['val_acc'] = val_acc
            sio.savemat('../../../results/' + datatask + '/crossval_ESN_Xy_ith_' + str(tt) + '_' + str(aa) + '.mat', results)