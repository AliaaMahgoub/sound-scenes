from sklearn.neural_network import MLPClassifier
from data_split import split_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

X = np.load('/home/aliaamahgoub/X.npy')
y = np.load('/home/aliaamahgoub/y.npy')
size_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]

X_ts, y_ts, X_train_folds, y_train_folds = split_data(X,y)

Nfolds = len(X_train_folds)
y = []
labels = []
#for num_layers in layers:
for size in size_list:
    print("current hidden size is: ", size)
    scores = []
    tr_scores = []

    for i in range(Nfolds):
        X_vl = X_train_folds[i]
        y_vl = y_train_folds[i]
        X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
        y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
        scaler = preprocessing.StandardScaler().fit(X_tr)
        X_tr_sc = scaler.transform(X_tr)
        X_vl_sc = scaler.transform(X_vl)
        clf = MLPClassifier(solver='lbfgs', max_iter=200,
                hidden_layer_sizes=(size))
        clf.fit(X_tr_sc, y_tr)
        scores.append(clf.score(X_vl_sc,y_vl))
        tr_scores.append(clf.score(X_tr_sc,y_tr))
    print("validation accuracy was ", np.mean(scores))
    print("training accuracy was ", np.mean(tr_scores))
    curr_lab = str(size)
    labels.append(curr_lab)
    y.append(np.mean(scores))

plt.rc('axes',axisbelow=True)
plt.grid()
plt.xlabel('hidden layer size')
plt.ylabel('CV accuracy')
plt.bar(labels,y,color='tab:blue')
plt.ylim([0,1])
plt.savefig("NN_cv_all.png")
plt.clf()
