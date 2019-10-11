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

Code:

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score, v_measure_score

    np.random.seed(233)

    digits = load_digits()
    data = scale(digits.data)   # 在均值附近集中化数据并缩放至单位方差
    reduced_data1 = PCA(n_components=2).fit_transform(data)
    reduced_data2 = TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(data))
    print(data.shape)

    n_samples, n_feature = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target


    # MeanShit
    X = data
    bandwidth = estimate_bandwidth(X, quantile=0.35, n_samples=n_samples)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels_ms = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels_ms)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    print('v_measure_score:', v_measure_score(labels, labels_ms))
    print('homogeneity_score:', homogeneity_score(labels, labels_ms))
    print('completeness_score:', completeness_score(labels, labels_ms))

    #############################################################################

    plt.figure(figsize=(10, 8))
    plt.clf()

    X1 = reduced_data1
    X2 = reduced_data2
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

    plt.subplot(121)
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels_ms == k
        cluster_center = cluster_centers[k]
        plt.plot(X1[my_members, 0], X1[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)

    plt.subplot(122)
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels_ms == k
        cluster_center = cluster_centers[k]
        plt.plot(X2[my_members, 0], X2[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)

    plt.show()

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

### Agglomerative Clustering
__整体过程基本和KMeans一样。__

聚类评价得分（数据类型、运行时间、NMI/V_MS、HS、CS）：

![图片9](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_6.jpg)

该算法的得分较Original KMeans有一些提升，较低于SpectralClustering的效果。

聚类可视化（从左到右-从上到下：原始数据分布下的真实标签(PCA可视化)、原始数据分布下的真实标签(PCA+t-SNE可视化)、原始数据分布下使用Agglomerative Clustering得到的标签(PCA可视化)、原始数据分布下使用Agglomerative Clustering得到的标签(PCA+t-SNE可视化)、PCA降维后的数据分布下使用Agglomerative Clustering得到的标签(PCA可视化)、PCA+t-SNE降维后的数据分布下使用Agglomerative Clustering得到的标签(PCA+t-SNE可视化)）：
![图片10](https://github.com/boceng/Data-Mining/blob/master/Homework1/Figure_5.png)

对方法参数linkage分别试了：ward、average、complete、single，除了ward外效果都非常差。

Code:

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score, v_measure_score

    np.random.seed(233)

    digits = load_digits()
    data = scale(digits.data)   # 在均值附近集中化数据并缩放至单位方差
    reduced_data1 = PCA(n_components=2).fit_transform(data)
    reduced_data2 = TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(data))
    print(data.shape)

    n_samples, n_feature = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target


    # AgglomerativeClustering
    y_predict_original = AgglomerativeClustering(n_clusters=n_digits, linkage='ward').fit_predict(data)
    y_predict_reduced = AgglomerativeClustering(n_clusters=n_digits, linkage='ward').fit_predict(reduced_data1)

    print('v_measure_score:', v_measure_score(labels, y_predict_original))
    print('homogeneity_score:', homogeneity_score(labels, y_predict_original))
    print('completeness_score:', completeness_score(labels, y_predict_original))
    print('----------')
    print('v_measure_score:', v_measure_score(labels, y_predict_reduced))
    print('homogeneity_score:', homogeneity_score(labels, y_predict_reduced))
    print('completeness_score:', completeness_score(labels, y_predict_reduced))


    # AgglomerativeClustering visualize
    plt.figure(figsize=(10, 10))

    plt.subplot(321)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=labels)
    plt.title('Ground Truth with PCA')

    plt.subplot(322)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=labels)
    plt.title('Ground Truth with PCA & TSNE')

    plt.subplot(323)
    y_predict_original = AgglomerativeClustering(n_clusters=n_digits, linkage='ward').fit_predict(data)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_original)
    plt.title('Original Features AgglomerativeClustering with PCA')

    plt.subplot(324)
    y_predict_original = AgglomerativeClustering(n_clusters=n_digits, linkage='ward').fit_predict(data)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_original)
    plt.title('Original Features AgglomerativeClustering with PCA & TSNE')

    plt.subplot(325)
    y_predict_reduced = AgglomerativeClustering(n_clusters=n_digits, linkage='ward').fit_predict(reduced_data1)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_reduced)
    plt.title('PCA-based Feature AgglomerativeClustering with PCA')

    plt.subplot(326)
    y_predict_reduced = AgglomerativeClustering(n_clusters=n_digits, linkage='ward').fit_predict(reduced_data2)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_reduced)
    plt.title('PCA & TSNE-based Feature AgglomerativeClustering with PCA & TSNE')

    plt.show()


### DBSCAN
__整体过程基本和KMeans一样。__

聚类评价得分（数据类型、运行时间、NMI/V_MS、HS、CS）：

![图片11](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_7.jpg)

很明显该聚类方法不适合手写数字辨识这个数据集，只有当minPts=1时，才能将完整性得分降至小于1，说明数据点之间欧式距离高度相近。

聚类可视化（从左到右-从上到下：原始数据分布下的真实标签(PCA可视化)、原始数据分布下的真实标签(PCA+t-SNE可视化)、原始数据分布下使用DBSCAN得到的标签(PCA可视化)、原始数据分布下使用DBSCAN得到的标签(PCA+t-SNE可视化)、PCA降维后的数据分布下使用DBSCAN得到的标签(PCA可视化)、PCA+t-SNE降维后的数据分布下使用DBSCAN得到的标签(PCA+t-SNE可视化)）：
![图片12](https://github.com/boceng/Data-Mining/blob/master/Homework1/Figure_6.png)

可视化效果同样也很差。

Code:

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score, v_measure_score

    np.random.seed(233)

    digits = load_digits()
    data = scale(digits.data)   # 在均值附近集中化数据并缩放至单位方差
    reduced_data1 = PCA(n_components=2).fit_transform(data)
    reduced_data2 = TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(data))
    print(data.shape)

    n_samples, n_feature = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target


    # DBSCAN
    y_predict_original = DBSCAN(eps=0.0001, min_samples=2).fit_predict(data)
    y_predict_reduced = DBSCAN(eps=0.0001, min_samples=2).fit_predict(reduced_data2)

    print('v_measure_score:', v_measure_score(labels, y_predict_original))
    print('homogeneity_score:', homogeneity_score(labels, y_predict_original))
    print('completeness_score:', completeness_score(labels, y_predict_original))
    print('------------------------------------')
    print('v_measure_score:', v_measure_score(labels, y_predict_reduced))
    print('homogeneity_score:', homogeneity_score(labels, y_predict_reduced))
    print('completeness_score:', completeness_score(labels, y_predict_reduced))


    # DBSCAN visualize
    plt.figure(figsize=(10, 10))

    plt.subplot(321)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=labels)
    plt.title('Ground Truth with PCA')

    plt.subplot(322)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=labels)
    plt.title('Ground Truth with PCA & TSNE')

    plt.subplot(323)
    y_predict_original = DBSCAN(eps=0.1, min_samples=1).fit_predict(data)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_original)
    plt.title('Original Features DBSCAN with PCA')

    plt.subplot(324)
    y_predict_original = DBSCAN(eps=0.1, min_samples=1).fit_predict(data)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_original)
    plt.title('Original Features DBSCAN with PCA & TSNE')

    plt.subplot(325)
    y_predict_reduced = DBSCAN(eps=0.1, min_samples=1).fit_predict(reduced_data1)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_reduced)
    plt.title('PCA-based Feature DBSCAN with PCA')

    plt.subplot(326)
    y_predict_reduced = DBSCAN(eps=0.1, min_samples=1).fit_predict(reduced_data2)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_reduced)
    plt.title('PCA & TSNE-based Feature DBSCAN with PCA & TSNE')

    plt.show()

### GaussianMixture
__整体过程基本和KMeans一样。__

聚类评价得分（数据类型、运行时间、NMI/V_MS、HS、CS）：

![图片13](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_8.jpg)

GMM应该能适合各种形状的分布，因为一般经过标准化后高斯分布往往能提供最佳的逼近，所以这里来看在手写数字辨识上的结果虽不是最优但也不是很差。

聚类可视化（从左到右-从上到下：原始数据分布下的真实标签(PCA可视化)、原始数据分布下的真实标签(PCA+t-SNE可视化)、原始数据分布下使用GaussianMixture得到的标签(PCA可视化)、原始数据分布下使用GaussianMixture得到的标签(PCA+t-SNE可视化)、PCA降维后的数据分布下使用GaussianMixture得到的标签(PCA可视化)、PCA+t-SNE降维后的数据分布下使用GaussianMixture得到的标签(PCA+t-SNE可视化)）：
![图片14](https://github.com/boceng/Data-Mining/blob/master/Homework1/Figure_7.png)

可视化也是发现并没有分得很糟，仅是一些不同的簇被分为同一簇了，比如5和2比较像这种情况。

Code:

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import scale
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score, v_measure_score

    np.random.seed(233)

    digits = load_digits()
    data = scale(digits.data)   # 在均值附近集中化数据并缩放至单位方差
    reduced_data1 = PCA(n_components=2).fit_transform(data)
    reduced_data2 = TSNE(n_components=2).fit_transform(PCA(n_components=50).fit_transform(data))
    print(data.shape)

    n_samples, n_feature = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target


    # GaussianMixture
    y_predict_original = GaussianMixture(n_components=n_digits).fit_predict(data)
    y_predict_reduced = GaussianMixture(n_components=n_digits).fit_predict(reduced_data1)

    print('v_measure_score:', v_measure_score(labels, y_predict_original))
    print('homogeneity_score:', homogeneity_score(labels, y_predict_original))
    print('completeness_score:', completeness_score(labels, y_predict_original))
    print('------------------------------------')
    print('v_measure_score:', v_measure_score(labels, y_predict_reduced))
    print('homogeneity_score:', homogeneity_score(labels, y_predict_reduced))
    print('completeness_score:', completeness_score(labels, y_predict_reduced))


    # GaussianMixture visualize
    plt.figure(figsize=(10, 10))

    plt.subplot(321)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=labels)
    plt.title('Ground Truth with PCA')

    plt.subplot(322)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=labels)
    plt.title('Ground Truth with PCA & TSNE')

    plt.subplot(323)
    y_predict_original = GaussianMixture(n_components=n_digits).fit_predict(data)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_original)
    plt.title('Original Features GaussianMixture with PCA')

    plt.subplot(324)
    y_predict_original = GaussianMixture(n_components=n_digits).fit_predict(data)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_original)
    plt.title('Original Features GaussianMixture with PCA & TSNE')

    plt.subplot(325)
    y_predict_reduced = GaussianMixture(n_components=n_digits).fit_predict(reduced_data1)
    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], c=y_predict_reduced)
    plt.title('PCA-based Feature GaussianMixture with PCA')

    plt.subplot(326)
    y_predict_reduced = GaussianMixture(n_components=n_digits).fit_predict(reduced_data2)
    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], c=y_predict_reduced)
    plt.title('PCA & TSNE-based Feature GaussianMixture with PCA & TSNE')

    plt.show()


## 手写数字聚类
### KMeans
__整体过程：得到原始数据通过opts判断是否进行归一化了，使用KMeans、MiniBatchKMeans分别对数据进行聚类，然后使用三种度量方法对其评估。__

聚类评价得分（NMI/V_MS、HS、CS）：

![图片15](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_9.jpg)

可见各项度量得分均偏低，说明tf-idf对文本映射到向量空间中丢失了很多信息，比如语义等信息。

观察使用tf-idf来映射文本时，进行聚类之后每个簇中前十个tf-idf权重最大的word：

![图片16](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_10.jpg)

### MiniBatchKMeans
__同样的过程得到MiniBatchKMeans的结果__

![图片17](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_11.jpg)

![图片18](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_12.jpg)

Code：

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn import metrics

    from sklearn.cluster import KMeans, MiniBatchKMeans

    import logging
    from optparse import OptionParser
    import sys
    from time import time

    import numpy as np


    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")

    # print(__doc__)
    # op.print_help()


    def is_interactive():
        return not hasattr(sys.modules['__main__'], '__file__')


    # work-around for Jupyter notebook and IPython console
    argv = [] if is_interactive() else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)


    # #############################################################################
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)

    print("%d documents" % len(dataset.data))
    print("%d categories" % len(dataset.target_names))
    print()

    labels = dataset.target
    true_k = np.unique(labels).shape[0]

    print("Extracting features from the training dataset "
          "using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset.data)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()


    # #############################################################################
    # Do the actual clustering

    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts.verbose)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    print()


    if not opts.use_hashing:
        print("Top terms per cluster:")

        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()

### AffinityPropagation
__整体过程：过程与kmeans大体一致。__

聚类评价得分（NMI/V_MS、HS、CS）：

![图片19](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_13.jpg)

完整性得分非常低，说明大部分相同簇的文本向量被分到不同的簇中了。

### MeanShift
__整体过程：过程与kmeans大体一致。__

聚类评价得分（NMI/V_MS、HS、CS）：

![图片20](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_14.jpg)

调了各种参数要么效果贼差，要么宕机。

### SpectralClustering
__整体过程：过程与kmeans大体一致。__

聚类评价得分（NMI/V_MS、HS、CS）：

![图片21](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_15.jpg)

同样地，完整性得分非常，说明很可能是映射到向量空间的方式不能保存足够的信息。

### AgglomerativeClustering
__整体过程：过程与kmeans大体一致。__

聚类评价得分（NMI/V_MS、HS、CS）：

![图片22](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_16.jpg)

相比前面的方法，这个方法得到分数比较稳定，但也不是非常高。

### DBSCAN
__整体过程：过程与kmeans大体一致。__

聚类评价得分（NMI/V_MS、HS、CS）：

![图片23](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_17.jpg)

同样完整性得分特别低，但也不排除DBSCAN算法并不适合去聚类文本数据。

### GaussianMixture
__整体过程：过程与kmeans大体一致。__

聚类评价得分（NMI/V_MS、HS、CS）：
![图片24](https://github.com/boceng/Data-Mining/blob/master/Homework1/result_18.jpg)

虽说GMM可以拟合各种数据分布，提供最佳的逼近，但也不是很会调参，留待以后解决吧。

All Code：

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn import metrics
    from sklearn.mixture import GaussianMixture

    from sklearn.cluster import *

    import logging
    from optparse import OptionParser
    import sys
    from time import time

    import numpy as np


    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")

    # print(__doc__)
    # op.print_help()


    def is_interactive():
        return not hasattr(sys.modules['__main__'], '__file__')


    # work-around for Jupyter notebook and IPython console
    argv = [] if is_interactive() else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)


    # #############################################################################
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)

    n_samples = len(dataset.data)
    print("%d documents" % n_samples)
    print("%d categories" % len(dataset.target_names))
    print()

    labels = dataset.target
    true_k = np.unique(labels).shape[0]

    print("Extracting features from the training dataset "
          "using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset.data)


    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()


    # #############################################################################
    # Do the actual clustering

    opts.minibatch = False
    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts.verbose)

    #AP
    km = AffinityPropagation(preference=-true_k)

    #MS
    # X = X.toarray()
    # bandwidth = estimate_bandwidth(X, quantile=0.35, n_samples=500)
    # km = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    #SC
    km = SpectralClustering(n_clusters=true_k)

    #AC
    km = AgglomerativeClustering(n_clusters=true_k, linkage='ward')
    # X = X.toarray()

    #DB
    km = DBSCAN(eps=0.5, min_samples=1)

    #GMM
    # km = GaussianMixture(n_components=true_k)
    # X = X.toarray()


    print("Clustering sparse data with %s" % km)
    t0 = time()
    lb = km.fit_predict(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, lb))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, lb))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, lb))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, lb))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, lb, sample_size=1000))

    print()


    if not opts.use_hashing:
        print("Top terms per cluster:")

        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()

