import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import os

n_neighbors = 5

path = '/home/aliaamahgoub/'
for weight in ['uniform', 'distance']:
    a = []
    b = []
    y = []
    for f in os.listdir(path):
        if f.endswith('.npy') and ((f.startswith('bandwidth')) or (f.startswith('centroid'))):
            x = np.load(path + f)
        if f.startswith('centroid'):
            a.append(x)
        elif f.startswith('bandwidth'):
            b.append(x)

            if 'airport' in f:
                y.append(np.array([0]*len(x)))
            elif 'shopping_mall' in f:
                y.append(np.array([1]*len(x)))
            elif 'metro_station' in f:
                y.append(np.array([2]*len(x)))
            elif 'street_pedestrian' in f:
                y.append(np.array([3]*len(x)))
            elif 'public_square' in f:
                y.append(np.array([4]*len(x)))
            elif 'street_traffic' in f:
                y.append(np.array([5]*len(x)))
            elif 'tram' in f:
                y.append(np.array([6]*len(x)))
            elif 'bus' in f:
                y.append(np.array([7]*len(x)))
            elif 'metro' in f:
                y.append(np.array([8]*len(x)))
            elif 'park' in f:
                y.append(np.array([9]*len(x)))
    a = np.hstack(a)
    b = np.hstack(b)
    y = np.hstack(y)
    X = np.vstack([a, b]).T
    h = 10
    #print(a.shape)
    #print(b.shape)
    #print(y.shape)
    #print(X.shape)

    #input()

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X,y)


    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    #plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z)

    sns.scatterplot(x=X[:, 0], y=X[:, 1],
        #palette=cmap_bold, alpha=1.0, edgecolor="black")
            alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title("10-Class classification (k = 5, weights = )",weight)
    plt.xlabel('Average Spectral Centroid')
    plt.ylabel('Average Spectral Bandwidth')
    if weight == 'distance':
        fig_name = "knn_sc_sb_distance.png"
    else:
        fig_name = "knn_sc_sb_uniform.png"
    plt.savefig(fig_name)
    plt.clf()

    a = []
    b = []
    y = []
    for f in os.listdir(path):
        if f.endswith('.npy') and ((f.startswith('centroid')) or (f.startswith('zc'))):
            x = np.load(path + f)
 
        if f.startswith('centroid'):
            a.append(x)
        elif f.startswith('zc'):
            b.append(x)

            if 'airport' in f:
                y.append(np.array([0]*len(x)))
            elif 'shopping_mall' in f:
                y.append(np.array([1]*len(x)))
            elif 'metro_station' in f:
                y.append(np.array([2]*len(x)))
            elif 'street_pedestrian' in f:
                y.append(np.array([3]*len(x)))
            elif 'public_square' in f:
                y.append(np.array([4]*len(x)))
            elif 'street_traffic' in f:
                y.append(np.array([5]*len(x)))
            elif 'tram' in f:
                y.append(np.array([6]*len(x)))
            elif 'bus' in f:
                y.append(np.array([7]*len(x)))
            elif 'metro' in f:
                y.append(np.array([8]*len(x)))
            elif 'park' in f:
                y.append(np.array([9]*len(x)))
    a = np.hstack(a)
    b = np.hstack(b)
    y = np.hstack(y)
    X = np.vstack([a, b]).T
    h = 0.5
    print(a.shape)
    print(b.shape)
    print(y.shape)
    print(X.shape)

    #input()

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X,y)


    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(x_min)
    print(x_max)
    print(y_min)
    print(y_max)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    #plt.contourf(xx, yy, Z, cmap=cmap_light)
    #print(xx.shape)
    #print(yy.shape)
    #print(Z.shape)
    plt.contourf(xx, yy, Z)

    sns.scatterplot(x=X[:, 0], y=X[:, 1],
        #palette=cmap_bold, alpha=1.0, edgecolor="black")
            alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title("10-Class classification (k = 5, weights = )",weight)
    plt.xlabel('Average Spectral Centroid')
    plt.ylabel('Average Zero Crossing Rate')
    if weight == 'distance':
        fig_name = "knn_sc_zcr_distance.png"
    else:
        fig_name = "knn_sc_zcr_uniform.png"
    plt.savefig(fig_name)
    plt.clf()

    a = []
    b = []
    y = []
    for f in os.listdir(path):
        if f.endswith('.npy') and ((f.startswith('centroid')) or (f.startswith('rms'))):
            x = np.load(path + f)

        if f.startswith('centroid'):
            a.append(x)
        elif f.startswith('rms'):
            b.append(x)

            if 'airport' in f:
                y.append(np.array([0]*len(x)))
            elif 'shopping_mall' in f:
                y.append(np.array([1]*len(x)))
            elif 'metro_station' in f:
                y.append(np.array([2]*len(x)))
            elif 'street_pedestrian' in f:
                y.append(np.array([3]*len(x)))
            elif 'public_square' in f:
                y.append(np.array([4]*len(x)))
            elif 'street_traffic' in f:
                y.append(np.array([5]*len(x)))
            elif 'tram' in f:
                y.append(np.array([6]*len(x)))
            elif 'bus' in f:
                y.append(np.array([7]*len(x)))
            elif 'metro' in f:
                y.append(np.array([8]*len(x)))
            elif 'park' in f:
                y.append(np.array([9]*len(x)))
    a = np.hstack(a)
    b = np.hstack(b)
    y = np.hstack(y)
    X = np.vstack([a, b]).T
    h = 0.5
    print(a.shape)
    print(b.shape)
    print(y.shape)
    print(X.shape)

    #input()

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X,y)


    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(x_min)
    print(x_max)
    print(y_min)
    print(y_max)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    #plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z)

    sns.scatterplot(x=X[:, 0], y=X[:, 1],
        #palette=cmap_bold, alpha=1.0, edgecolor="black")
            alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title("10-Class classification (k = 5, weights = )",weight)
    plt.xlabel('Average Spectral Centroid')
    plt.ylabel('Average Root Mean Square Energy')
    if weight == 'distance':
        fig_name = "knn_sc_rms_distance.png"
    else:
        fig_name = "knn_sc_rms_uniform.png"
    plt.savefig(fig_name)
    plt.clf()

    a = []
    b = []
    y = []
    for f in os.listdir(path):
        if f.endswith('.npy') and ((f.startswith('bandwidth')) or (f.startswith('zc'))):
            x = np.load(path + f)
        if f.startswith('bandwidth'):
            a.append(x)
        elif f.startswith('zc'):
            b.append(x)

            if 'airport' in f:
                y.append(np.array([0]*len(x)))
            elif 'shopping_mall' in f:
                y.append(np.array([1]*len(x)))
            elif 'metro_station' in f:
                y.append(np.array([2]*len(x)))
            elif 'street_pedestrian' in f:
                y.append(np.array([3]*len(x)))
            elif 'public_square' in f:
                y.append(np.array([4]*len(x)))
            elif 'street_traffic' in f:
                y.append(np.array([5]*len(x)))
            elif 'tram' in f:
                y.append(np.array([6]*len(x)))
            elif 'bus' in f:
                y.append(np.array([7]*len(x)))
            elif 'metro' in f:
                y.append(np.array([8]*len(x)))
            elif 'park' in f:
                y.append(np.array([9]*len(x)))
    a = np.hstack(a)
    b = np.hstack(b)
    y = np.hstack(y)
    X = np.vstack([a, b]).T
    h = 0.5
    #print(a.shape)
    #print(b.shape)
    #print(y.shape)
    #print(X.shape)

    #input()

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X,y)


    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(x_min)
    print(x_max)
    print(y_min)
    print(y_max)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    #plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z)

    sns.scatterplot(x=X[:, 0], y=X[:, 1],
        #palette=cmap_bold, alpha=1.0, edgecolor="black")
            alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title("10-Class classification (k = 5, weights = )",weight)
    plt.xlabel('Average Spectral Bandwidth')
    plt.ylabel('Average Zero Crossing Rate')
    if weight == 'distance':
        fig_name = "knn_sb_zc_distance.png"
    else:
        fig_name = "knn_sb_zc_uniform.png"
    plt.savefig(fig_name)
    plt.clf()


    a = []
    b = []
    y = []
    for f in os.listdir(path):
        if f.endswith('.npy') and ((f.startswith('bandwidth')) or (f.startswith('rms'))):
            x = np.load(path + f)
        if f.startswith('bandwidth'):
            a.append(x)
        elif f.startswith('rms'):
            b.append(x)

            if 'airport' in f:
                y.append(np.array([0]*len(x)))
            elif 'shopping_mall' in f:
                y.append(np.array([1]*len(x)))
            elif 'metro_station' in f:
                y.append(np.array([2]*len(x)))
            elif 'street_pedestrian' in f:
                y.append(np.array([3]*len(x)))
            elif 'public_square' in f:
                y.append(np.array([4]*len(x)))
            elif 'street_traffic' in f:
                y.append(np.array([5]*len(x)))
            elif 'tram' in f:
                y.append(np.array([6]*len(x)))
            elif 'bus' in f:
                y.append(np.array([7]*len(x)))
            elif 'metro' in f:
                y.append(np.array([8]*len(x)))
            elif 'park' in f:
                y.append(np.array([9]*len(x)))
    a = np.hstack(a)
    b = np.hstack(b)
    y = np.hstack(y)
    X = np.vstack([a, b]).T
    h = 0.5
    #print(a.shape)
    #print(b.shape)
    #print(y.shape)
    #print(X.shape)

    #input()

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X,y)


    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    #plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z)

    sns.scatterplot(x=X[:, 0], y=X[:, 1],
        #palette=cmap_bold, alpha=1.0, edgecolor="black")
            alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title("10-Class classification (k = 5, weights = )",weight)
    plt.xlabel('Average Spectral Bandwidth')
    plt.ylabel('Average Root Mean Square Energy')
    if weight == 'distance':
        fig_name = "knn_sb_rms_distance.png"
    else:
        fig_name = "knn_sb_rms_uniform.png"
    plt.savefig(fig_name)
    plt.clf()
    a = []
    b = []
    y = []
    for f in os.listdir(path):
        if f.endswith('.npy') and ((f.startswith('rms')) or (f.startswith('zc'))):
            x = np.load(path + f)
        if f.startswith('zc'):
            a.append(x)
        elif f.startswith('rms'):
            b.append(x)

            if 'airport' in f:
                y.append(np.array([0]*len(x)))
            elif 'shopping_mall' in f:
                y.append(np.array([1]*len(x)))
            elif 'metro_station' in f:
                y.append(np.array([2]*len(x)))
            elif 'street_pedestrian' in f:
                y.append(np.array([3]*len(x)))
            elif 'public_square' in f:
                y.append(np.array([4]*len(x)))
            elif 'street_traffic' in f:
                y.append(np.array([5]*len(x)))
            elif 'tram' in f:
                y.append(np.array([6]*len(x)))
            elif 'bus' in f:
                y.append(np.array([7]*len(x)))
            elif 'metro' in f:
                y.append(np.array([8]*len(x)))
            elif 'park' in f:
                y.append(np.array([9]*len(x)))
    a = np.hstack(a)
    b = np.hstack(b)
    y = np.hstack(y)
    X = np.vstack([a, b]).T
    h = 0.5
    #print(a.shape)
    #print(b.shape)
    #print(y.shape)
    #print(X.shape)

    #input()

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'c', 'darkblue']
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X,y)


    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    #plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z)

    sns.scatterplot(x=X[:, 0], y=X[:, 1],
        #palette=cmap_bold, alpha=1.0, edgecolor="black")
            alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title("10-Class classification (k = 5, weights = )",weight)
    plt.xlabel('Average Zero Crossing Rate')
    plt.ylabel('Average Root Mean Square Energy')
    if weight == 'distance':
        fig_name = "knn_zc_rms_distance.png"
    else:
        fig_name = "knn_zc_rms_uniform.png"
    plt.savefig(fig_name)
    plt.clf()
