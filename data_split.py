import numpy as np

#X = np.load('/home/aliaamahgoub/X.npy')
#y = np.load('/home/aliaamahgoub/y.npy')

def split_data(X, y, Nfolds = 5):
    Ntotal = X.shape[0] 
    Ntest = int(Ntotal/10)
    Ntrain = Ntotal - Ntest

    total_idx = np.random.choice(range(Ntotal),Ntotal,replace=False)
    test_idx = total_idx[:int(Ntest)]
    X_train_folds = []
    y_train_folds = []
    
    train_idx = np.delete(total_idx, test_idx)
    for i in range(Nfolds):
        temp_idx = train_idx[i*int(Ntrain/Nfolds):i*int(Ntrain/Nfolds)+int(Ntrain/Nfolds)]
        '''
        train_idx = np.random.choice(temp, Ntrain, replace=False)
        train_idx = train_idx[:int(Ntrain/Nfolds)]
        np.append(test_idx, train_idx)
        '''
        X_train_folds.append(X[temp_idx])
        y_train_folds.append(y[temp_idx])

    X_ts = X[test_idx]
    y_ts = y[test_idx]
    '''
    print('shape of X_ts ', X_ts.shape)
    print('shape of y_ts ', y_ts.shape)
    #print(X_train_folds)
    #print(y_train_folds)
    print(len(X_train_folds))
    print(len(y_train_folds))
    print(X_train_folds[1].shape)
    print(y_train_folds[1].shape)
    '''
    return X_ts, y_ts, X_train_folds, y_train_folds

#split_data(X, y)
    # X_ts is np array w shape Ntest, 4
    # y_ts is a np array w shape Ntest
    # X_train_folds is a list w Nfolds elements, where each element is a np array w a fifth of the training data
    # y_train_folds is a list w Nfolds elements, where each element is a np array w a fifth of the training data
