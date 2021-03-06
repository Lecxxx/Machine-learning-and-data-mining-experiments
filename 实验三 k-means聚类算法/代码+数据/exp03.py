import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为1000，默认为50
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 修改RC参数，让其可以支持中文
plt.rcParams['font.sans-serif']=['SimHei']

data=pd.read_table('D:\learn\大三上\机器学习\实验三\代码+数据\Iris\iris.data',sep=',',header=None,encoding='ansi')
# print(data) # 查看数据 可以看到一共有多少种class

# 将iris class的数据类型转换为数值型 便于后面的计算
print('\n','class类型转换后的数据：')
data.loc[data[4]=='Iris-setosa',4]=0
data.loc[data[4]=='Iris-versicolor',4]=1
data.loc[data[4]=='Iris-virginica',4]=2
print(data) # 查看转换后的结果

# print(arr)
# iris = load_iris()
# print(iris)
# X = iris.data   # 特征向量，并且是按顺序排列的
# lable = iris.target  # 标签
# # print('\n',X)
# # print('\n',lable)

# 数据集预处理
arr=np.array(data.iloc[:,:-1])
sepal_area= arr[:,0]*arr[:,1] # 以花萼面积为横坐标
petal_area= arr[:,2]*arr[:,3] # 以花瓣面积做纵坐标
class_list=data.loc[:,4]
# print(sepal_area)
# print(petal_area)
# print(class_list)

# 从样本集中随机选取k个样本作为簇中心
def randomSelect(k):
    temp =set() # 因为不允许再一次随机选择中选到重复的点 所以利用set来实现 在set中不能有重复的key
    while(len(temp)<k):
        temp.add(np.random.randint(0,150))
    return(temp)

# 计算欧式距离 (x,y)到(x0,y0)的距离
def distance(x1,y1,x2,y2,k):
    '''
    计算欧氏距离
    欧式距离计算公式为：((x2-x1)^2+(y2-y1)^2)^(1/2)
    '''
    ds = [[]for i in range(len(x))] # ds为二维数组 x为所有的样本点的花萼面积
    i = 0
    j = 0
    while i < len(x):
        while j < k: # 遍历k个簇中心
            M = np.sqrt(np.power((x1[i] - x2[j]), 2) + np.power((y1[i] - y2[j]), 2)) # 计算所有样本点的花萼面积（x）与花瓣面积（y）和各个簇中心(共三个)的距离
            M = round(M,1) # 计算出的欧氏距离保留一位小数
            j = j + 1
            ds[i].append(M) # 将欧式距离存入列表ds中
        j = 0
        i = i + 1
    return(ds)

# 计算样本点到簇中心的距离 用于判断簇中心是否发生移动
def  distanceCenter(x3,y3,x4,y4,k):
    i = 0
    j = 0
    sum = 0
    while i < k: # 遍历k个簇中心
        while j < len(x): # 遍历所有样本点
            M = np.power((x3[j] - x4[i]), 2) + np.power((y3[j] - y4[i]), 2)
            M = round(M,1)
            sum += M
            j = j + 1
        j = 0
        i = i + 1
    return(sum)

# 计算各个簇的中心点
def centerPoint(temp,k):
    mean_x = [] # 记录横坐标的平均值
    mean_y = [] # 记录纵坐标的平均值
    i = 0
    j = 0
    while i < k: # 共有k个簇
        sum_x = 0 # 第i个簇所有横坐标的总和
        sum_y = 0 # 第i个簇所有纵坐标的总和
        count = 0
        while j < len(x): # 遍历所有样本点
            if i == temp[j]: # 根据新划分好的簇 计算各个簇的簇中心
                count = count + 1
                sum_x = sum_x + x[j]
                sum_y = sum_y + y[j]
            j = j + 1
        # 第i个簇新的簇中心
        center_x = sum_x / count # 第i个簇所有横坐标的平均值
        center_y = sum_y / count # 第i个簇所有纵坐标的平均值
        mean_x.append(center_x)
        mean_y.append(center_y)
        j = 0
        i = i + 1
    return[mean_x,mean_y]

# 按照k值聚类
def kmeans(ds,lenth):
    i = 0
    temp = []
    while i < lenth: # 遍历所有样本点
        temp.append(ds[i].index(min(ds[i]))) # 找到每个点与簇中心之间欧式距离最短的簇中心的下标 即对于每一个样本 将其划分到与其距离最近的簇中心所在的簇中
        i = i + 1
    return(temp)





# 先从样本集中随机选取 k个样本作为簇中心 并计算所有样本与这 k个“簇中心”的距离 对于每一个样本将其划分到与其距离最近的“簇中心”所在的簇中 对于新的簇计算各个簇的新的“簇中心”
'''
根据以上描述，我们大致可以猜测到实现kmeans算法的主要四点：
（1）簇个数 k 的选择
（2）各个样本点到“簇中心”的距离
（3）根据新划分的簇，更新“簇中心”
（4）重复上述2、3过程，直至"簇中心"没有移动
'''
#这里聚3类 k取3是因为class只有3类 分别为Iris Setosa、Iris Versicolour和Iris Virginica
k = 3

Random_dot = randomSelect(k)
# print(Random_dot)
# 将随机选择到的k个簇中心的相关数据记录下来 用于后续的测试
sepal_test = [sepal_area[i] for i in range(len(sepal_area)) if (i in Random_dot)]
petal_test = [petal_area[i] for i in range(len(petal_area)) if (i in Random_dot)]
class_text = [class_list[i] for i in range(len(class_list)) if (i in Random_dot)]
# print(sepal_test)
# print(petal_test)
# print(class_text)

#(x,y)为所有样本点的坐标 (x0,y0)为簇中心的坐标
x = sepal_area # 以所有样本点的花萼面积为横坐标
y = petal_area # 以所有样本点的花瓣面积做纵坐标
x0 = sepal_test # 以随机选择到的k个簇中心的花萼面积为横坐标
y0 = petal_test # 以随机选择到的k个簇中心的花瓣面积为纵坐标

n = 0 # 记录迭代的次数 初始时的簇中心为利用随机函数所选取到的
ds = distance(x,y,x0,y0,k)
# print(ds)
temp_kmeans = kmeans(ds,len(x))
# print(temp_kmeans)
temp_center = distanceCenter(x,y,x0,y0,k)

n = n + 1 # 第一次迭代
center = centerPoint(temp_kmeans,k) # 对于新的簇计算各个簇的新的簇中心
# print(center)
x0 = center[0] # 新的簇中心的横坐标
# print(x0)
y0 = center[1] # 新的簇中心的纵坐标
# print(y0)
ds = distance(x,y,x0,y0,k)
temp_kmeans = kmeans(ds,len(x))
temp_iteration = distanceCenter(x,y,x0,y0,k) # 迭代后各个样本点与新的簇中心的距离

# 比较前后两次与簇中心的距离 判断是否相等 不相等说明新的簇中心发生了移动 继续迭代 直至新的簇中心没有移动
while np.abs(temp_iteration - temp_center) != 0: # abs()为计算绝对值
    temp_center = temp_iteration
    center = centerPoint(temp_kmeans,k)
    x0 = center[0]
    y0 = center[1]
    ds = distance(x,y,x0,y0,k)
    temp_kmeans = kmeans(ds,len(x))
    temp_iteration = distanceCenter(x,y,x0,y0,k)
    n = n + 1

#结果可视化
print("迭代次数: ", n) # 统计出迭代次数
print('簇中心位置：',x0,y0)
plt.scatter(x0,y0,color='r',s=50,marker='s')
plt.scatter(x,y,c=temp_kmeans,s=25,marker='o')
plt.xlabel('花萼面积')
plt.ylabel('花瓣面积')
plt.title("聚3类")
plt.show()

