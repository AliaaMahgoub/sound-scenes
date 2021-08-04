from data_split import split_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
  
print("RMS & ZC")

X  = np.load('/home/aliaamahgoub/X_rms_zc.npy')
y = np.load('/home/aliaamahgoub/y_rms_zc.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    print(weight,'\n')

    y = []
    yerr = []

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.grid()
    plt.ylim([0,1])
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("RMS and ZCR -", weight))
    if weight == 'uniform':
        plt.savefig("knn_cv_rms_zc_uniform.png")
    else:
        plt.savefig("knn_cv_rms_zc_distance.png")

    plt.clf()


print("ZC\n")

X  = np.load('/home/aliaamahgoub/X_zc.npy')
y = np.load('/home/aliaamahgoub/y_zc.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []
    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.grid()
    plt.ylim([0,1])
    plt.ylabel("accuracy (out of 1)")
    plt.title(("ZCR -", weight))
    if weight == 'uniform':
        plt.savefig("knn_cv_zc_uniform.png")
    else:
        plt.savefig("knn_cv_zc_distance.png")

    plt.clf()

print("RMS\n")

X  = np.load('/home/aliaamahgoub/X_rms.npy')
y = np.load('/home/aliaamahgoub/y_rms.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    print(weight,'\n')
    y = []
    yerr = []

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("RMS - ", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_rms_uniform.png")
    else:
        plt.savefig("knn_cv_rms_distance.png")

    plt.clf()

print("SC, ZC, RMS\n")

X  = np.load('/home/aliaamahgoub/X_c_zc_rms.npy')
y = np.load('/home/aliaamahgoub/y_c_zc_rms.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []
    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.grid()
    plt.ylim([0,1])
    plt.title(("SC, ZCR, RMS -", weight))
    if weight == 'uniform':
        plt.savefig("knn_cv_c_zc_rms_uniform.png")
    else:
        plt.savefig("knn_cv_c_zc_rms_distance.png")
    plt.clf()
 

print("SC & ZC\n")

X  = np.load('/home/aliaamahgoub/X_c_zc.npy')
y = np.load('/home/aliaamahgoub/y_c_zc.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []
    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SC and ZCR -", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_c_zc_uniform.png")
    else:
        plt.savefig("knn_cv_c_zc_distance.png")

    plt.clf()

print("SC & RMS\n")

X  = np.load('/home/aliaamahgoub/X_c_rms.npy')
y = np.load('/home/aliaamahgoub/y_c_rms.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []
    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.grid()
    plt.ylim([0,1])
    plt.title(("SC and RMS -", weight))
    if weight == 'uniform':
        plt.savefig("knn_cv_c_rms_uniform.png")
    else:
        plt.savefig("knn_cv_c_rms_distance.png")

    plt.clf()

print("SC, SB, ZC\n")

X  = np.load('/home/aliaamahgoub/X_c_b_zc.npy')
y = np.load('/home/aliaamahgoub/y_c_b_zc.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []
    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SC, SB, ZCR -", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_c_b_zc_uniform.png")
    else:
        plt.savefig("knn_cv_c_b_zcdistance.png")

    plt.clf()

print("SC, SB, RMS\n")

X  = np.load('/home/aliaamahgoub/X_c_b_rms.npy')
y = np.load('/home/aliaamahgoub/y_c_b_rms.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []

    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SC, SB, RMS -", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_c_b_rms_uniform.png")
    else:
        plt.savefig("knn_cv_c_b_rms_distance.png")
    plt.clf()
 
print("SC & SB\n")

X  = np.load('/home/aliaamahgoub/X_c_b.npy')
y = np.load('/home/aliaamahgoub/y_c_b.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []

    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SC and SB -", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_c_b_uniform.png")
    else:
        plt.savefig("knn_cv_c_b_distance.png")

    plt.clf()

print("SC\n")

X  = np.load('/home/aliaamahgoub/X_c.npy')
y = np.load('/home/aliaamahgoub/y_c.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []

    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SC", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_c_uniform.png")
    else:
        plt.savefig("knn_cv_c_distance.png")

    plt.clf()

print("SB, ZC, RMS\n")

X  = np.load('/home/aliaamahgoub/X_b_zc_rms.npy')
y = np.load('/home/aliaamahgoub/y_b_zc_rms.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []

    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SB, ZC, RMS -", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_b_zc_rms_uniform.png")
    else:
        plt.savefig("knn_cv_b_zc_rms_distance.png")
    plt.clf()
 

print("SB & ZC\n")

X  = np.load('/home/aliaamahgoub/X_b_zc.npy')
y = np.load('/home/aliaamahgoub/y_b_zc.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []

    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SB and ZCR -", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_b_zc_uniform.png")
    else:
        plt.savefig("knn_cv_b_zc_distance.png")

    plt.clf()

print("SB and RMS\n")

X  = np.load('/home/aliaamahgoub/X_b_rms.npy')
y = np.load('/home/aliaamahgoub/y_b_rms.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []
    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SB and RMS -", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_b_rms_uniform.png")
    else:
        plt.savefig("knn_cv_b_rms_distance.png")

    plt.clf()

print("SB\n")

X  = np.load('/home/aliaamahgoub/X_b.npy')
y = np.load('/home/aliaamahgoub/y_b.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []

    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("SB - ", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_b_uniform.png")
    else:
        plt.savefig("knn_cv_b_distance.png")

    plt.clf()

print("all features\n")

X  = np.load('/home/aliaamahgoub/X.npy')
y = np.load('/home/aliaamahgoub/y.npy')
k_list = [1, 5, 21, 51, 121, 221, 299, 421, 599, 799, 999]
#annotations = [str(a) for a in k_list]
X_ts, y_ts, X_train_folds, y_train_folds = split_data(X, y)

for weight in ['uniform', 'distance']:
    
    y = []
    yerr = []

    print(weight,'\n')

    for k in k_list:
        Nfolds = len(X_train_folds)
    
        scores = []

        for i in range(Nfolds):
            X_vl = X_train_folds[i]
            y_vl = y_train_folds[i]
            X_tr = np.vstack([X_train_folds[j] for j in range(Nfolds) if j != i])
            y_tr = np.hstack([y_train_folds[j] for j in range(Nfolds) if j != i])
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_tr, y_tr)
            scores.append(knn.score(X_vl, y_vl))
    
        print(k, ' : ', round(np.mean(scores), 6),'\n')
        #print(np.std(scores))
        y.append(np.mean(scores))
        yerr.append(np.std(scores))

    plt.errorbar(k_list, y, yerr=yerr)
    plt.xlabel("K")
    plt.ylabel("accuracy (out of 1)")
    plt.title(("all features - ", weight))
    plt.grid()
    plt.ylim([0,1])
    if weight == 'uniform':
        plt.savefig("knn_cv_all_uniform.png")
    else:
        plt.savefig("knn_cv_all_distance.png")
    plt.clf()
