# 聚类
## 手写数字聚类
### KMeans
__整体过程：得到原始数据先进行预处理，将数据在均值附近集中化并缩放至单位误差，使用KMeans分别对预处理后的数据、PCA降维后的数据、PCA+t-SNE降维后的数据进行聚类，然后使用三种度量方法对其评估，并对得到的簇进行可视化。__

聚类评价得分（数据类型、运行时间、样本到最近簇中心平方和、NMI/V_MS、HS、CS）：
![图片1](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_1.jpg)

可以意外发现通过PCA+t-SNE降维后的数据特征进行聚类要比直接对原始数据特征的各个度量都要高出很多的得分。

聚类可视化（从左到右-从上到下：原始数据分布下的真实标签(PCA可视化)、原始数据分布下的真实标签(PCA+t-SNE可视化)、原始数据分布下使用KMeans得到的标签(PCA可视化)、原始数据分布下使用KMeans得到的标签(PCA+t-SNE可视化)、PCA降维后的数据分布下使用KMeans得到的标签(PCA可视化)、PCA+t-SNE降维后的数据分布下使用KMeans得到的标签(PCA+t-SNE可视化)）：
![图片2](https://github.com/boceng/Data-Mining/blob/master/Homework1/Figure_1.png)

意外发现PCA+t-SNE对数据降维后非常切合Ground truth的分布。

Code：

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
    from sklearn.metrics import completeness_score, v_measure_score
    from time import time

    np.random.seed(233)

    digits = load_digits()
    data = scale(digits.data)   # 在均值附近集中化数据并缩放至单位方差
    reduced_data1 = PCA(n_components=2).fit_transform(data)
    reduced_data2 = TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(data))
    print(data.shape)

    n_samples, n_feature = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target


    def bench_estimate(estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f'
              % (name, (time() - t0), estimator.inertia_,   # 样本到其最近簇的平方距离之和
                 v_measure_score(labels, estimator.labels_),
                 homogeneity_score(labels, estimator.labels_),
                 completeness_score(labels, estimator.labels_)))


    bench_estimate(KMeans(n_clusters=n_digits, n_init=10), name="Ordinary data", data=data)
    bench_estimate(KMeans(n_clusters=n_digits, n_init=10), name="PCA-Reduced data", data=reduced_data1)
    bench_estimate(KMeans(n_clusters=n_digits, n_init=10), name="PCA & TSNE-Reduced data", data=reduced_data2)


    plt.figure(figsize=(12, 12))

    plt.subplot(321)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=labels)
    plt.title('Ground Truth with PCA')

    plt.subplot(322)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=labels)
    plt.title('Ground Truth with PCA & TSNE')

    plt.subplot(323)
    y_predict_original = KMeans(n_clusters=n_digits, n_init=10).fit_predict(data)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_original)
    plt.title('Original Features KMeans with PCA')

    plt.subplot(324)
    y_predict_original = KMeans(n_clusters=n_digits, n_init=10).fit_predict(data)
    plt.scatter(reduced_data2[:, 0], reduced_data1[:, 1], c=y_predict_original)
    plt.title('Original Features KMeans with PCA & TSNE')

    plt.subplot(325)
    y_predict_reduced = KMeans(n_clusters=n_digits, n_init=10).fit_predict(reduced_data1)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_reduced)
    plt.title('PCA-based Feature KMeans with PCA')

    plt.subplot(326)
    y_predict_reduced = KMeans(n_clusters=n_digits, n_init=10).fit_predict(reduced_data2)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_reduced)
    plt.title('PCA & TSNE-based Feature KMeans with PCA & TSNE')

    # print(normalized_mutual_info_score(labels, y_predict_original))
    # print(homogeneity_score(labels, y_predict_original))
    # print(completeness_score(labels, y_predict_original))
    #
    # print(normalized_mutual_info_score(labels, y_predict_reduced))
    # print(homogeneity_score(labels, y_predict_reduced))
    # print(completeness_score(labels, y_predict_reduced))

    plt.show()

### KMeans++、MiniBatchKMeans

聚类评价得分（数据类型、运行时间、样本到最近簇中心平方和、NMI/V_MS、HS、CS）：
![图片3](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_2.jpg)

比较上述两种Kemans变体方法的得分可知：KMeans++通过对数据的初始化的改进很明显提高了一部分得分；MiniBatchKMeans相对于KMeans的得分要差上一些，但算法运行速度提升了非常多。

### AffinityPropagation
__整体过程基本和KMeans一样，不过采用了不同的可视化方法。__

聚类评价得分（数据类型、运行时间、NMI/V_MS、HS、CS）：
![图片4](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_3.jpg)

该算法的得分较KMeans有的度量有一些提升，有的度量有一些下降（包括原始数据特征和降维后的数据特征）。

聚类可视化（从左到右-从上到下：原始数据分布下使用AffinityPropagation得到的标签(PCA可视化)、原始数据分布下使用AffinityPropagation得到的标签(PCA+t-SNE可视化)、PCA降维后的数据分布下使用AffinityPropagation得到的标签(PCA可视化)、PCA+t-SNE降维后的数据分布下使用AffinityPropagation得到的标签(PCA+t-SNE可视化)）：
![图片5](https://github.com/boceng/Data-Mining/blob/master/Homework1/Figure_2.png)

会发现，很明显使用原始数据特征进行聚类的效果比较好，但该算法对降维后的数据特征进行聚类基本坏掉了（调了半天参数仍没找好的结果）。

Code：

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
    from sklearn.metrics import completeness_score, v_measure_score
    from time import time

    np.random.seed(233)

    digits = load_digits()
    data = scale(digits.data)   # 在均值附近集中化数据并缩放至单位方差
    reduced_data1 = PCA(n_components=2).fit_transform(data)
    reduced_data2 = TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(data))
    print(data.shape)

    # print(data.mean(axis=0))
    # print(data.std(axis=0))

    n_samples, n_feature = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target


    def bench_estimate2(estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f'
              % (name, (time() - t0),
                 v_measure_score(labels, estimator.labels_),
                 homogeneity_score(labels, estimator.labels_),
                 completeness_score(labels, estimator.labels_)))


    # AffinityPropagation
    bench_estimate2(AffinityPropagation(preference=-3100), name="AffinityPropagation Ordinary data", data=data)
    bench_estimate2(AffinityPropagation(preference=-1145), name="AffinityPropagation PCA-Reduced data",
                    data=reduced_data1)
    bench_estimate2(AffinityPropagation(preference=-1145), name="AffinityPropagation PCA & TSNE-Reduced data",
                    data=reduced_data2)

    af1 = AffinityPropagation(preference=-3100).fit(data)
    af2 = AffinityPropagation(preference=-1145).fit(reduced_data1)
    af3 = AffinityPropagation(preference=-1145).fit(reduced_data2)

    plt.close('all')
    plt.figure(figsize=(11, 11))
    plt.clf()
    X1 = reduced_data1
    X2 = reduced_data2
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

    plt.subplot(221)
    cluster_centers_indices = af1.cluster_centers_indices_
    labels_ = af1.labels_
    n_clusters_ = len(cluster_centers_indices)

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels_ == k
        cluster_center = X1[cluster_centers_indices[k]]
        plt.plot(X1[class_members, 0], X1[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X1[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('Original Feature with PCA\nEstimated number of clusters: %d' % n_clusters_)

    plt.subplot(222)

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels_ == k
        cluster_center = X2[cluster_centers_indices[k]]
        plt.plot(X2[class_members, 0], X2[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X2[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('Original Feature with PCA & t-SNE\nEstimated number of clusters: %d' % n_clusters_)

    plt.subplot(223)
    cluster_centers_indices = af2.cluster_centers_indices_
    labels_ = af2.labels_
    n_clusters_ = len(cluster_centers_indices)

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels_ == k
        cluster_center = X1[cluster_centers_indices[k]]
        plt.plot(X1[class_members, 0], X1[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X1[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('PCA-based Feature with PCA\nEstimated number of clusters: %d' % n_clusters_)

    plt.subplot(224)
    cluster_centers_indices = af3.cluster_centers_indices_
    labels_ = af2.labels_
    n_clusters_ = len(cluster_centers_indices)

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels_ == k
        cluster_center = X2[cluster_centers_indices[k]]
        plt.plot(X2[class_members, 0], X2[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X2[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('PCA & t-SNE-based Feature with PCA & t-SNE\nEstimated number of clusters: %d' % n_clusters_)

    plt.show()

### MeanShift
__整体过程如上。__

聚类评价得分：

![图片6](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_4.jpg)

只是简单调了一下估计带宽的方法参数，0-1都试了下均得不到好的结果，待解决。

聚类可视化（从左到右-从上到下：原始数据分布下使用MeanShift得到的标签(PCA可视化)、原始数据分布下使用MeanShift得到的标签(PCA+t-SNE可视化)）：
![图片7](https://github.com/boceng/Data-Mining/blob/master/Homework1/Figure_3.png)

结果很明显坏掉了。。

### SpectralClustering
__先通过KNN得到相似性矩阵，然后利用拉普拉斯矩阵得到特征矩阵作为最终feature。__

聚类评价得分（数据类型、运行时间、NMI/V_MS、HS、CS）：

![图片7](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_5.jpg)

该算法的得分较Original KMeans要普遍好一些。

聚类可视化（从左到右-从上到下：原始数据分布下使用SpectralClustering得到的标签(PCA可视化)、原始数据分布下使用SpectralClustering得到的标签(PCA+t-SNE可视化)、原始数据分布下使用SpectralClustering得到的标签(PCA可视化)、原始数据分布下使用SpectralClustering得到的标签(PCA+t-SNE可视化)）：
![图片8](https://github.com/boceng/Data-Mining/blob/master/Homework1/Figure_4.png)

如图4，很明显它较groundtruth的同质性得分应该还是比降维后的数据使用KMeans低一些的。

Code:

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation
    from sklearn.cluster import MeanShift, estimate_bandwidth, SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
    from sklearn.metrics import completeness_score, v_measure_score
    from time import time

    np.random.seed(233)

    digits = load_digits()
    data = scale(digits.data)   # 在均值附近集中化数据并缩放至单位方差
    reduced_data1 = PCA(n_components=2).fit_transform(data)
    reduced_data2 = TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(data))
    print(data.shape)

    # print(data.mean(axis=0))
    # print(data.std(axis=0))

    n_samples, n_feature = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target

    # SpectralClustering

    X = data

    def distance(p1,p2):  # 欧式距离
        return np.linalg.norm(p1-p2)
    def getWbyKNN(data,k):  # 利用KNN获得相似度矩阵
        points_num = len(data)
        dis_matrix = np.zeros((points_num,points_num))
        W = np.zeros((points_num,points_num))
        for i in range(points_num):
            for j in range(i+1,points_num):
                dis_matrix[i][j] = dis_matrix[j][i] = distance(data[i],data[j])
        for idx,each in enumerate(dis_matrix):
            index_array  = np.argsort(each)
            W[idx][index_array[1:k+1]] = 1  # 距离最短的是自己
        tmp_W = np.transpose(W)
        W = (tmp_W+W)/2  #转置相加除以2是为了让矩阵对称
        return W
    def getD(W):    # 获得度矩阵
        points_num = len(W)
        D = np.diag(np.zeros(points_num))
        for i in range(points_num):
            D[i][i] = sum(W[i])
        return D
    def getEigVec(L,cluster_num):  #从拉普拉斯矩阵获得特征矩阵
        eigval,eigvec = np.linalg.eig(L)
        dim = len(eigval)
        dictEigval = dict(zip(eigval,range(0,dim)))
        kEig = np.sort(eigval)[0:cluster_num]
        ix = [dictEigval[k] for k in kEig]
        return eigval[ix], eigvec[:,ix]


    cluster_num = n_digits
    KNN_k = 5
    W = getWbyKNN(X,KNN_k)
    D = getD(W)
    L = D-W
    eigval, eigvec = getEigVec(L, cluster_num)

    print(np.shape(eigvec))

    sc = SpectralClustering(n_clusters=cluster_num).fit(eigvec)
    labels_sc = sc.labels_
    print('v_measure_score:', v_measure_score(labels, labels_sc))
    print('homogeneity_score:', homogeneity_score(labels, labels_sc))
    print('completeness_score:', completeness_score(labels, labels_sc))

    plt.figure(figsize=(11, 8))

    plt.subplot(221)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=labels)
    plt.title('Ground Truth with PCA')

    plt.subplot(222)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=labels)
    plt.title('Ground Truth with PCA & TSNE')

    plt.subplot(223)
    y_predict_original = SpectralClustering(n_clusters=n_digits, n_init=10).fit_predict(eigvec)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_original)
    plt.title('Original Features SpectralClustering with PCA')

    plt.subplot(224)
    y_predict_original = SpectralClustering(n_clusters=n_digits, n_init=10).fit_predict(eigvec)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_original)
    plt.title('Original Features SpectralClustering with PCA & TSNE')

    plt.show()


### other
