# coding=gbk
# coding: utf-8
'''
Created on Jul 12, 2014
python 科学计算学习：numpy快速处理数据测试
@author: 皮皮
'''
# import string
import matplotlib.pyplot as plt  

if __name__ == '__main__':    
    fp = open(r"E:\machine_learning\datasets\housing_data\housing_data_years_price.txt", 'r')
    linesList = fp.readlines()
#     print(linesList)
    linesList = [line.strip().split(" ") for line in linesList]
    fp.close()    
    
    print("linesList:")
    print(linesList)
#     years = [string.atof(x[0]) for x in linesList]
    years = [x[0] for x in linesList]
    print(years)
    price = [x[1] for x in linesList]
    print(price)
    
    plt.plot(years, price, 'answers*')#,label="$cos(x^2)$")
    plt.plot(years, price, 'r')
#     plt.plot(years, price, 'o')    #散点图
    plt.xlabel("years(+2000)")
    plt.xlim(0, )
    plt.ylabel("housing average price(*2000 yuan)")
    plt.ylim(0, )
    plt.title('line_regression & gradient decrease')
#     plt.legend()
    plt.show()
