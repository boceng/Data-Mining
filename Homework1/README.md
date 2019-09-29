# 聚类
## 手写数字聚类
### KMeans
__整体过程：得到原始数据先进行预处理，将数据在均值附近集中化并缩放至单位误差，使用KMeans分别对预处理后的数据、PCA降维后的数据、PCA+t-SNE降维后的数据进行聚类，然后使用三种度量方法对其评估，并对得到的簇进行可视化。__

聚类评价得分（数据类型、运行时间、样本到最近簇中心平方和、NMI/V_MS、HS、CS）：
![图片1](https://github.com/boceng/Data-Mining/blob/master/Homework1/TIM%E6%88%AA%E5%9B%BE20190929104144.jpg)

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

### other
