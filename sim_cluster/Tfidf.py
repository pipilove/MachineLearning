#!/usr/bin/env python
# coding=gbk
"""
__title__ = 'TF-IDF algorithm'
__author__ = 'pi'
__mtime__ = '2014.12.20'
#code is far away from bug with the god animal protecting
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
import numpy


def tfidf(term, doc, docset):
    """
    calculate the TF-IDF of a term in doc_line
    :param term: string
    :param doc: str list
    :param docset: str list
    :return:tfidf float
    """
    # tf = float(该词在文件中的出现次数) / 在文件中所有字词的出现次数之和
    tf = float(doc.count(term)) / sum(doc.count(word) for word in set(doc))  #set(doc_line)前是否要清洗？
    #idf = log( 总文件数目 / (包含该词语之文件的数目+1) )
    idf = numpy.log(float(len(docset)) / ( len([doc for doc in docset if term in doc]) + 1 ))  #每个文档中都存在就成负数了
    #idf = scipy.log(...)
    if idf < 0.0:  #或者+1去掉
        idf = 0.0
    return tf * idf


def driver_test():
    """
    tfidf()测试驱动程序
    :return:
    """
    """
    :return:
    """
    a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
    docset = [a, abb, abc]
    print(tfidf("a", a, docset))
    print(tfidf("b", abb, docset))
    print(tfidf("a", abc, docset))
    print(tfidf("b", abc, docset))
    print(tfidf("c", abc, docset))


if __name__ == '__main__':
    driver_test()
