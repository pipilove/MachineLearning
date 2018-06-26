#!/usr/bin/env python
# coding=gbk
"""
__title__ = 'TF-IDF algorithm'
__author__ = 'pi'
__mtime__ = '2014.12.20'
#code is far away from bug with the god animal protecting
              ��������������
            �����ߩ��������ߩ�
            ������������������
            �����ש������ס���
            ���������ߡ�������
            ������������������
            ����������������������
            �����������ޱ��ӡ�  �ǩ�
            ������������BUG�� ������
            ���������������ש�����
            ���������ϩϡ����ϩ�
            ���������ߩ������ߩ�
"""
import numpy


def tfidf(term, doc, docset):
    """
    calculate the TF-IDF of prefers term in doc_line
    :param term: string
    :param doc: str list
    :param docset: str list
    :return:tfidf float
    """
    # tf = float(�ô����ļ��еĳ��ִ���) / ���ļ��������ִʵĳ��ִ���֮��
    tf = float(doc.count(term)) / sum(doc.count(word) for word in set(doc))  #set(doc_line)ǰ�Ƿ�Ҫ��ϴ��
    #idf = log( ���ļ���Ŀ / (�����ô���֮�ļ�����Ŀ+1) )
    idf = numpy.log(float(len(docset)) / ( len([doc for doc in docset if term in doc]) + 1 ))  #ÿ���ĵ��ж����ھͳɸ�����
    #idf = scipy.log(...)
    if idf < 0.0:  #����+1ȥ��
        idf = 0.0
    return tf * idf


def driver_test():
    """
    tfidf()������������
    :return:
    """
    """
    :return:
    """
    a, abb, abc = ["prefers"], ["prefers", "answers", "answers"], ["prefers", "answers", "c"]
    docset = [a, abb, abc]
    print(tfidf("prefers", a, docset))
    print(tfidf("answers", abb, docset))
    print(tfidf("prefers", abc, docset))
    print(tfidf("answers", abc, docset))
    print(tfidf("c", abc, docset))


if __name__ == '__main__':
    driver_test()
