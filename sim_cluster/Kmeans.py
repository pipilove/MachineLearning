#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'flat Clustering – kmeans plot'
__author__ = 'pi'
__mtime__ = '2014.12.16'
#code is far away from bugs with the god animal protecting
                 ┏┓　　　┏┓
               ┏┛┻━━━┛┻┓
               ┃　　　━　　　┃
               ┃　┳┛　┗┳　┃
               ┃　　　┻　　　┃
               ┗━┓　　　┏━┛
               　　┃　　　┗━━━┓
               　　┃　神兽保佑　  ┣┓
               　　┃　永无BUG！ 　┏┛
               　　┗┓┓┏━┳┓┏┛
               　　　┃┫┫　┃┫┫
               　　　┗┻┛　┗┻┛
"""
import pdb
from sklearn import datasets
from sklearn.cluster import KMeans

from RelatedPosts import StemmedTfidfVectorizer, norm_euclidDist

"""
loading datasets into train_dataset_bunch
"""
MLCOMP_DIR = r'./datasets'
groups = ['comp.graphics', 'comp.os.ms-windows.misc']   #仅用于小数据测试时用
# groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.ma c.hardware', 'comp.windows.x', 'sci.space']  # 中量数据时用
train_dataset_bunch = datasets.load_mlcomp(name_or_id='20news-18828', set_='train', mlcomp_root=MLCOMP_DIR,categories=groups)  #<class 'sklearn.datasets.base.Bunch'>
# print(type(train_dataset_bunch.data)) #list
# print(type(train_dataset_bunch.data[0]))  #bytes
# print(len(train_dataset_bunch.filenames)) # #posts=3414


"""
fit_transform dataset.data into train_dataset_mats
"""
Vectorizer = StemmedTfidfVectorizer
vectorizer = Vectorizer(decode_error='ignore', stop_words='english', min_df=10, max_df=0.5)
# scipy.sparse.csr.csr_matrix
train_dataset_mats = vectorizer.fit_transform(train_dataset_bunch.data)
num_samples, num_features = train_dataset_mats.shape
print("#samples:%d, #features:%d" % (num_samples, num_features))

"""
clustering
"""
num_clusters = 50
km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
km.fit(train_dataset_mats)  # clustered =
print("\n#km.class_labels:", km.labels_.shape, "\nkm.labels_[0:10]:", km.labels_[0:10])
# print(len(set(km.labels_)))   #total 50 diff class_labels
print("#km.cluster_centers_:\n", km.cluster_centers_.shape, "\nkm.cluster_centers_[:1]", km.cluster_centers_[:1], "\n")

"""
metrics for kmeans clustering
"""
print("metrics:")
labels = train_dataset_bunch.target
from sklearn import metrics

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand Index: %0.3f" %
      metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted Mutual Information: %0.3f" %
      metrics.adjusted_mutual_info_score(labels, km.labels_))
print(("Silhouette Coefficient: %0.3f\n" %
       metrics.silhouette_score(train_dataset_mats, labels, sample_size=1000)))

"""
LanguageAnalysis test_post
"""
test_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.After 1 year it is working only sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks."""
test_data_mat = vectorizer.transform([test_post])
test_post_labels = km.predict(test_data_mat)  # ndarray
test_post_label = test_post_labels[0]
# test_post_label不重要，可以不一样，但对应的similarity, post应该一样就行
print("test_post_label:", test_post_label)

"""
find similar docs with test_post (in one cluster)
"""
similar_indices = (km.labels_ == test_post_label).nonzero()[0]  #nonzero returns tuple
# print(len(similar_indices))
test_similar_docs = []  #tuple list
for i in similar_indices:
    dist = norm_euclidDist(test_data_mat, train_dataset_mats[i])
    test_similar_docs.append((dist, train_dataset_bunch.data[i]))
test_similar_docs = sorted(test_similar_docs)
# test_similar_docs.sort()
print("#test_similar_docs: ", len(test_similar_docs))
# pdb.set_trace()
print(test_similar_docs[0])  #(similarity, post_str)tuple    most similar post
# print test_similar_docs[0][0],test_similar_docs[0][1][:100]
# print(len(test_similar_docs[0][1]))   # #char in post_str
