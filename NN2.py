from sklearn.neural_network import MLPClassifier
from data_split import split_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

X = np.load('/home/aliaamahgoub/X.npy')
y = np.load('/home/aliaamahgoub/y.npy')
l1_size_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
#l1_size_list = [2, 4, 8]
l2_size_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
#l2_size_list = [2, 4, 8]

X_ts, y_ts, X_train_folds, y_train_folds = split_data(X,y)

Nfolds = len(X_train_folds)
#yerr = []
y = []
labels = []
#j = 1

#for num_layers in layers:
for size_1 in l1_size_list:
    for size_2 in l2_size_list:
        print("current hidden size is: ", size_1, size_2)
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
            clf = MLPClassifier(solver='lbfgs', max_iter=300,
                    hidden_layer_sizes=(size_1, size_2))
            clf.fit(X_tr_sc, y_tr)
            scores.append(clf.score(X_vl_sc,y_vl))
            tr_scores.append(clf.score(X_tr_sc,y_tr))
        print("validation accuracy was ", np.mean(scores))
        print("training accuracy was ", np.mean(tr_scores))
        #j += 1
        curr_lab = str(size_1)+','+str(size_2)
        labels.append(curr_lab)
        y.append(np.mean(scores))
        #yerr.append(np.std(scores))

#plt.errorbar(size_list, y, yerr=yerr)
plt.rc('axes', axisbelow=True)
plt.grid()
plt.xlabel('hidden layer sizes (layer 1, layer 2)')
plt.bar(labels ,y, color = 'tab:cyan')
plt.ylabel('CV accuracy')
plt.ylim([0,1])
plt.savefig("NN2_cv_all.png")
plt.clf()
