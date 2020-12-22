import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib; matplotlib.use('TkAgg')
from sklearn import preprocessing
import os
import sys
from pandas.core.frame import DataFrame
from exp01 import avg,std,consGrade #此处调用了上一个实验中计算平均值、标准差以及转换体能测试成绩为数值型的方法

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为1000，默认为50
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)

#-----------------------------------------------请以课程1成绩为x轴，体能成绩为y轴，画出散点图-----------------------------------------------
df = pd.read_csv('D:\python_work\机器学习实验二\objectFile.csv')
plt.title('Scatter Diagram') #设置标题
plt.xlabel('C1-grade') #设置横坐标名称
plt.ylabel('Constitution-grade') #设置纵坐标名称
my_x_ticks = np.arange(0, 100, 1) #分数区间为[0,100]，故设定原始数据有100个点，设置从0开始，间隔为1
plt.xticks(my_x_ticks)
plt.scatter(df['C1'], df['Constitution']) #画出散点图
plt.show() #将散点图显示出来

#-----------------------------------------------以5分为间隔，画出课程1的成绩直方图-----------------------------------------------
df = pd.read_csv('D:\python_work\机器学习实验二\objectFile.csv')
df['C1'].hist(color = 'green') #绘制课程1成绩的直方图，将直方图颜色改为绿色
plt.title('Histogram') #设置标题
plt.ylabel('Population') #设置纵坐标名称
plt.xlabel('C1-grade') #设置横坐标名称
my_x_ticks = np.arange(0, 100, 5) #分数区间为[0,100]，故设定原始数据有100个点，设置从0开始，间隔为5
plt.xticks(my_x_ticks)
plt.show()

#-----------------------------------------------对每门成绩进行z-score归一化，得到归一化的数据矩阵-----------------------------------------------
df = pd.read_csv('D:\python_work\机器学习实验二\objectFile.csv')
df['Constitution']=df['Constitution'].apply(consGrade) #将体测成绩进行转换，便于后面的计算
#需要进行z-score归一化操作的属性名
attributeList = df.columns.values.tolist()[5:16]
#对每门成绩进行z-score归一化
for i in attributeList:
    tempList = df[i].tolist() #生成一个该属性下元素的列表
    avg_vale = avg(df[i]) #平均值
    std_vale = std(df[i], avg(df[i])) #方差
    for j in range(len(tempList)):
        if  std_vale == 0: #标准差为0直接跳过 如数据中的属性“C10”列
            continue
        else:
            temp = (tempList[j] - avg_vale) / std_vale #将数据按期属性（按列进行）减去其均值，并处以其方差，得到的结果是对于每个属性/每列来说所有数据都聚集在0附近，方差为1
        df.loc[j,i] = temp #loc取单行单列，即为某一个坐标的值，覆盖掉数据中原有的值

#找出数据中的每门成绩进行z-score归一化的相关信息
zScore = df.loc[:,'C1':'Constitution']
zScore.index = [np.array(df.loc[:,'ID'])] #列属性为学生的ID号
print('打印归一化的数据矩阵：\n',zScore)
#将实验结果导入一个csv文件中保存下来
zScore.to_csv('zscoreFile.csv',header=True)

#-----------------------------------------------计算出100x100的相关矩阵，并可视化出混淆矩阵-----------------------------------------------
df = pd.read_csv('D:\python_work\机器学习实验二\objectFile.csv')
df['Constitution']=df['Constitution'].apply(consGrade) # 将体测成绩进行转换，便于后面的计算
df['C6'] = df['C6'].apply(lambda x:x*10)
df['C7'] = df['C7'].apply(lambda x:x*10)
df['C8'] = df['C8'].apply(lambda x:x*10)
df['C9'] = df['C9'].apply(lambda x:x*10)
# print(df)

attributeList = df.columns.values.tolist()[5:14] # 此处的相关性只用考虑属性为成绩的 并且由于实验数据中C10项均为0 所以不纳入计算的访问中
attributeList.append('Constitution')
# print(attributeList)

avg_list=[] # 存储每个学生成绩的平均值
std_list=[] # 存储每个学生成绩的标准差

# 计算每个学生成绩的平均值
for i in range(len(df)):
    sum=0 # 学生成绩总和
    for j in attributeList:
        sum+=df.loc[i,j]
    avg_list.append(sum/len(attributeList))

#计算每个学生成绩的样本标准差
for i in range(len(df)):
    count=0
    for j in attributeList:
        count+=((df.loc[i,j]-avg_list[i])**2)
    std_list.append((count/(len(attributeList)-1))**0.5)

# print(avg_list)
# print(std_list)

# 计算两个学生之间的协方差
def Corvar(i,j):
    temp_i=[] # 记录i学生某门成绩减去其对应平均值
    temp_j=[] # 记录j学生某门成绩减去其对应平均值
    count = 0
    result = 0
    for column in attributeList:
        temp_i.append(df.loc[i,column]-avg_list[i])
        temp_j.append(df.loc[j,column]-avg_list[j])
    for index in range(len(attributeList)):
        count += temp_i[index]*temp_j[index]
    result = count/(len(attributeList)-1)
    return result

# 生成一个n*n的相关矩阵 其中n为学生样本数目
correlationMatrix = pd.DataFrame(data=np.arange((len(df))**2).reshape(len(df),len(df)))

for i in range(len(df)):
    for j in range(len(df)):
        correlationMatrix.loc[i,j]=Corvar(i, j)/(std_list[i]*std_list[j])

# correlationMatrix中的行属性名和列属性名均为学生的ID号
# print(np.array(df.loc[:,'ID']))
correlationMatrix.columns = [df.loc[:,'ID']]
correlationMatrix.index = [np.array(df.loc[:,'ID'])]
print('打印出91x91的相关矩阵\n',correlationMatrix)
# 将实验结果导入一个csv文件中保存下来
correlationMatrix.to_csv('correlationMatrixFile.csv',header=True)

# 画热力图
dfData = correlationMatrix.corr()
# print(dfData)
plt.subplots(figsize=(len(df), len(df))) # 设置画面大小
sns.heatmap(data =dfData, # 指定绘图数据
            linewidths = .1, # 设置每个单元格边框的宽度
            cmap="Blues", # 指定填充色
            square=True #意思是将图变成一个正方形，默认是一个矩形
            )
plt.show()

#-----------------------------------------------根据相关矩阵，找到距离每个样本最近的三个样本，得到100x3的矩阵（每一行为对应三个样本的ID）输出到txt文件中，以\t,\n间隔-----------------------------------------------
df = pd.read_csv('D:\python_work\机器学习实验二\objectFile.csv')
correlationMatrix = pd.read_csv('D:\python_work\机器学习实验二\correlationMatrixFile.csv')
Matrix = pd.DataFrame(data=np.arange(len(df)*3).reshape(len(df),3))

# correlationMatrix=correlationMatrix.sort_values(by=0,ascending=False,axis=1)  # 按第一行降序排序
# print(correlationMatrix)
# Matrix中的列属性名均为学生的ID号
Matrix.index = [df.loc[:,'ID']]
# print(Matrix)
# print(names[1])

for i in range(len(correlationMatrix)):
    correlationMatrix = correlationMatrix.sort_values(by=i, ascending=False, axis=1) # 按第i行降序排序
    names = correlationMatrix.columns.tolist()  # 获取相关矩阵的列名
    index=0
    count=2 # 若元素为自身 不算入其中
    while index < 3: # 找到其中元素除自身以外最大的三个 易知第一个元素为自身与自身的相关系数
        Matrix.iloc[i, index] = names[count]
        index+=1
        count+=1

print('输出处理后的矩阵：\n',Matrix)
Matrix.to_csv('D:/python_work/机器学习实验二/Matrix.txt',sep='\t',index=False,header=None)