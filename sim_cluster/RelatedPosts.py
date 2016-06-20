#!/usr/bin/env python
# coding=gbk
"""
__title__ = 'Clustering – Finding Related Posts'
__author__ = 'pi'
__mtime__ = '2014.12.3'
#code is far away from bugs with the god animal protecting
              ┏┓　　　┏┓
            ┏┛┻━━━┛┻┓
            ┃　　　?　　　┃
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
import os
import sys
import scipy as sp
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# stemmer for stop_words in english
english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):
    """
    stem the posts before we feed them into CountVectorizer.overwrite the method build_analyzer.
    """

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    stem the posts before we feed them into TfidfVectorizer.overwrite the method build_analyzer.
    """

    def build_analyzer(self):  # 注释掉即成为普通TfidfVectorizer
        # analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        analyzer = TfidfVectorizer.build_analyzer(self)
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def convert():
    """
    Converting raw text into a bag-of-words & Counting words
    """
    DIR = r'E:\mine\python_workspace\datasets\Willi Richert-1400OS_Code/1400OS_03_Codes/data/toy'
    posts = [open(os.path.join(DIR, file)).read() for file in os.listdir(DIR)]  # file str list
    print("training posts : ",posts)
    # for post in posts: print(post)
    # print()
    exit()

    Vectorizer = StemmedTfidfVectorizer
    vectorizer = Vectorizer(stop_words='english', min_df=1)
    # print sorted(countVectorizer.get_stop_words())[0:10]
    post_train_mats = vectorizer.fit_transform(posts)  # scipy.sparse.csr.csr_matrix
    # print(post_train_mats)                                  #sparse mat
    print("all features' name : ")
    print(vectorizer.get_feature_names())  # stemmed之后不一定是正确的有意义的单词
    print()

    print("all training posts arrays : ")
    print(post_train_mats.toarray())  # dense mat
    # print(post_train_mats.getcol(4).toarray())
    post_num, feature_num = post_train_mats.shape  # dense mat's shape
    print("#post_num=%d\t\t#feature_num=%d" % (post_num, feature_num))
    print()

    new_post = 'imaging databases'
    print("LanguageAnalysis post : ", repr(new_post))
    new_post_mat = vectorizer.transform([new_post])
    print(new_post_mat.toarray())
    print()

    return new_post_mat, new_post, post_train_mats, posts


def norm_euclidDist(new_post_mat, post_train_mat):
    """
    calculate norm_euclidDist between 2 mats
    :param:
    """
    # try:
    normal_new_post = new_post_mat / sp.linalg.norm(new_post_mat.toarray())
    normal_post_train = post_train_mat / sp.linalg.norm(post_train_mat.toarray())
    # except(ZeroDivisionError):
    # return sys.maxint
    # print normal_post_train
    # if normal_new_post.shape != normal_post_train.shape:
    # print "inconsistent shapes: normal_new_post.shape != normal_post_train.shape!!!"
    # exit()
    delta = normal_new_post - normal_post_train
    return sp.linalg.norm(delta.toarray())


def select_most_similar(new_post_mat, new_post, post_train_mats, posts):
    """
    select the best similar post with the new post:
    """
    min_dist = sys.maxint
    best_i = None
    best_post = None
    for i in range(0, post_train_mats.shape[0]):
        if posts[i] == new_post:
            continue
        post_train_mat = post_train_mats[i]  # .getrow(i)

        dist = norm_euclidDist(new_post_mat, post_train_mat)
        print("post %d is similar to test_post with %.2f" % (i, dist))
        if dist < min_dist:
            min_dist = dist
            best_i = i
    print("post %d is most similar to the new_post_mat with %.2f : %s" % (best_i, min_dist, posts[i]))


if __name__ == '__main__':
    new_post_mat, new_post, post_train_mats, posts = convert()
    select_most_similar(new_post_mat, new_post, post_train_mats, posts)
