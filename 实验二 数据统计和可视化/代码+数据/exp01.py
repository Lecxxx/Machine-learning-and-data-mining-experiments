import pandas as pd
import random
import numpy as np
import os
import glob
import csv

# #显示所有列
# pd.set_option('display.max_columns', None)
# #显示所有行
# pd.set_option('display.max_rows', None)
# #设置value的显示长度为1000，默认为50
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', 1000)
#
# #------------------------------------------合并文件------------------------------------------
#
# #列出当前目录下所有的文件
# files = os.listdir('.')
# for filename in files:
# 	portion = os.path.splitext(filename)
# 	#如果后缀是.txt
# 	if portion[1] == ".txt":
# 		#重新组合文件名和后缀名,修改为csv格式
# 		newname = portion[0] + ".csv"
# 		os.rename(filename,newname)
#
# df1=pd.read_excel('D:\python_work\机器学习实验一\一.数据源1'+ '.xlsx',index_col=0) #index_col=0表示以原有数据的第一列(索引为0)当作行索引
# df1.to_csv(r'D:\python_work\机器学习实验一\一.数据源1'+ '.csv', encoding='utf_8_sig') #将excel文件转化为csv文件
#
# df2=pd.read_csv('D:\python_work\机器学习实验一\一.数据源2-逗号间隔'+ '.csv',index_col=0)
# df2.to_csv(r'D:\python_work\机器学习实验一\一.数据源2-逗号间隔'+ '.csv')
#
# print(u'文件格式转换完毕')
# csv_list = glob.glob('*.csv') #查看同文件夹下的csv文件数
# print(u'共发现 %s 个CSV文件'% len(csv_list))
# print(u'正在进行合并处理......')
# for i in csv_list: #循环读取同文件夹下的csv文件
#     fr = open(i,'r').read()
#     with open('tempFile.csv','a') as f: #将结果保存为result.csv
#     	f.write(fr)
# print(u'合并完毕！')
# f.close()
#
# #------------------------------------------数据清洗------------------------------------------
# output_csv_path = r'D:\python_work\机器学习实验一\tempFile.csv'
# df=pd.read_csv(output_csv_path) #如果指定了列名header=None;如果没有指定列名，默认header=0
# print('合并后的文件为：\n',df)
#
# #标准化性别
# Gender=df['Gender'].tolist() #列表的元素也可以是列表
# Gender_new=[]
# for i in Gender:
#     if i=='female':
#         Gender_new.append('girl')
#     if i=='male':
#         Gender_new.append('boy')
#     if i=='boy':
#         Gender_new.append('boy')
#     if i=='girl':
#         Gender_new.append('girl')
#     if i=='Gender':
#         Gender_new.append('Gender')
# df['Gender']=Gender_new
#
# #找到缺失元素并对符合要求的赋值
# def fillVacancy(column,df):
#     column_list=df[df[column].isnull()].index.tolist() #用于判断缺失值并找到其下标索引
#     #isnull()返回了布尔值。该处为缺失值，返回True；该处不为缺失值，则返回False
#     for i in range(len(column_list)):
#         for j in range(len(df)):
#             if df.Name[j] == df.Name[column_list[i]] and j != column_list:
#                 df.loc[df.index[column_list[i]], column] = df.loc[df.index[j], column] #将合并后同一项的数据中查询到的已知值赋值给缺失值
#                 #用loc取出需要填充的行标签和列标签
#
# fillVacancy('ID',df)
# fillVacancy('Name',df)
# fillVacancy('City',df)
# fillVacancy('Gender',df)
# fillVacancy('Height',df)
# fillVacancy('C1',df)
# fillVacancy('C2',df)
# fillVacancy('C3',df)
# fillVacancy('C4',df)
# fillVacancy('C5',df)
# fillVacancy('C6',df)
# fillVacancy('C7',df)
# fillVacancy('C8',df)
# fillVacancy('C9',df)
#
# #用于填充缺失值
# df['C10']=df['C10'].fillna(0) #将缺失值填补为0
# df['Constitution']=df['Constitution'].fillna('unknown') #将Constitution缺失值填补为unknow
# df=df.fillna(0) #将数据中所有NAN值转换为0
#
# #标准化身高
# df['Height'] = df['Height'].apply(lambda x:float(x)*100 if x!='Height' and float(x)<100.0 else x)
#
# #标准化ID号
# def standardId(x):
#     if len(str(x))==1:
#         return '20200'+str(x)
#     elif x != 'ID' and len(str(x))==2:
#         return '2020'+str(x)
#     elif len(str(x)) ==3:
#         return '202'+str(x)
#     else:
#         return str(x)
#
# df['ID'] = df['ID'].apply(standardId)
#
# #将数据转换成float，便于后续去重
# df['C1'] = df['C1'].apply(lambda x:float(x) if x!='C1' else x)
# df['C2'] = df['C2'].apply(lambda x:float(x) if x!='C2' else x)
# df['C3'] = df['C3'].apply(lambda x:float(x) if x!='C3' else x)
# df['C4'] = df['C4'].apply(lambda x:float(x) if x!='C4' else x)
# df['C5'] = df['C5'].apply(lambda x:float(x) if x!='C5' else x)
# df['C6'] = df['C6'].apply(lambda x:float(x) if x!='C6' else x)
# df['C7'] = df['C7'].apply(lambda x:float(x) if x!='C7' else x)
# df['C8'] = df['C8'].apply(lambda x:float(x) if x!='C8' else x)
# df['C9'] = df['C9'].apply(lambda x:float(x) if x!='C9' else x)
#
# #去除重复行
# df.to_csv('objectFile.csv',index=False,header=True)
# df=pd.read_csv('objectFile.csv',header=None) #如果指定了列名header=None;如果没有指定列名，默认header=0
# df.drop_duplicates(inplace=True) #inplace=True表示直接在原来的DataFrame上删除重复项，而默认值False表示生成一个副本
# df=df.sort_values(by=0,ascending=False,axis=0)
# df=df.reset_index(drop=True) #重置行索引
#
# df.to_csv('objectFile.csv',index=False,header=None)
#
# df=pd.read_csv('objectFile.csv',header=0)
#
# #只要检测到ID相同的则判定为数据失效，将所有ID相同的数据项全部删除
# temp=[]
# def dedupId(df):
#     for i in range(0,len(df)):
#         for j in range(i+1,len(df)):
#             if df.ID[i] == df.ID[j]:
#                 temp.append(i)
#                 temp.append(j)
#
# dedupId(df)
# for i in range(len(temp)):
#     df.drop(index=temp[i], axis=0 ,inplace=True) #删除其中ID相同的行
#
# df=df.reset_index(drop=True)#重置行索引
# print('\n数据清洗已完成,结果如下所示：\n',df)
# df.to_csv('objectFile.csv',index=False,header=True)
#
# #------------------------------------------学生中家乡在Beijing的所有课程的平均成绩------------------------------------------
# df_BJ=df[df['City'] == 'Beijing'] #找出家乡是北京的所有数据项

#平均值公式
def avg(df_avg):
    count = 0
    result =0
    for i in range(len(df_avg)):
        count += df_avg[df_avg.index[i]]
    result = count/len(df_avg)
    return result

# C1_avg=avg(df_BJ['C1'])
# C2_avg=avg(df_BJ['C2'])
# C3_avg=avg(df_BJ['C3'])
# C4_avg=avg(df_BJ['C4'])
# C5_avg=avg(df_BJ['C5'])
# C6_avg=avg(df_BJ['C6'])
# C7_avg=avg(df_BJ['C7'])
# C8_avg=avg(df_BJ['C8'])
# C9_avg=avg(df_BJ['C9'])
# C10_avg=avg(df_BJ['C10'])
#
# print('\n北京学生C1课程平均成绩：%.2f'% C1_avg)
# print('北京学生C2课程平均成绩：%.2f'% C2_avg)
# print('北京学生C3课程平均成绩：%.2f'% C3_avg)
# print('北京学生C4课程平均成绩：%.2f'% C4_avg)
# print('北京学生C5课程平均成绩：%.2f'% C5_avg)
# print('北京学生C6课程平均成绩：%.2f'% C6_avg)
# print('北京学生C7课程平均成绩：%.2f'% C7_avg)
# print('北京学生C8课程平均成绩：%.2f'% C8_avg)
# print('北京学生C9课程平均成绩：%.2f'% C9_avg)
# print('北京学生C10课程平均成绩：%.2f'% C10_avg)
#
# #------------------------------------------学生中家乡在广州，课程1在80分以上，且课程9在9分以上的男同学的数量------------------------------------------
# df_GZ_boy=df[(df['City'] == 'Guangzhou') & (df['C1']>=80) & (df['C9']>9) & (df['Gender']=='boy')] #找出家乡是广州、课程1在80分以上、课程9在9分以上且性别是男的的所有数据项
# print('\n广州男学生C1成绩80分以上并且C9成绩9分以上数量为：%.2f'%len(df_GZ_boy))
#
# #------------------------------------------比较广州和上海两地女生的平均体能测试成绩，哪个地区的更强些------------------------------------------
# df_GZ_girl=df[(df['City']=='Guangzhou')&(df['Gender']=='girl')] #找出家乡分别是广州和北京的且性别为女孩的所有数据项
# df_SH_girl=df[(df['City']=='Shanghai')&(df['Gender']=='girl')]

#转换成绩为数值型，便于后面的计算
def consGrade(x):
    if x=='excellent':
        return 95
    elif x=='good':
        return 85
    elif x=='general':
        return 75
    elif x=='bad':
        return 60
    elif x=='unknown':
        return 0

# df_GZ_girl=df_GZ_girl['Constitution'].apply(consGrade)
# df_SH_girl=df_SH_girl['Constitution'].apply(consGrade)
# df_GZ_girlavg=avg(df_GZ_girl)
# df_SH_girlavg=avg(df_SH_girl)
# print('\n广州女生的平均体能测试成绩为：',df_GZ_girlavg)
# print('上海女生的平均体能测试成绩为：',df_SH_girlavg)
#
# if df_GZ_girlavg > df_SH_girlavg:
#     print('广州女生的平均体能测试成绩比上海女生的好')
# elif df_SH_girlavg > df_GZ_girlavg:
#     print('上海女生的平均体能测试成绩比广州女生的好')
# else:
#     print('广州女生的平均体能测试成绩与上海女生的相同')
#
# #------------------------------------------学习成绩和体能测试成绩，两者的相关性是多少------------------------------------------
#
# avg_temp=[] #存放平均值
# #计算各门成绩的平均值
# Constitution=df['Constitution'].apply(consGrade)#将体测成绩进行转换 便于后面的计算
# C1_avg=avg(df['C1'])
# # C2_avg=avg(df['C2'])
# # C3_avg=avg(df['C3'])
# # C4_avg=avg(df['C4'])
# # C5_avg=avg(df['C5'])
# # C6_avg=avg(df['C6'])
# # C7_avg=avg(df['C7'])
# # C8_avg=avg(df['C8'])
# # C9_avg=avg(df['C9'])
# # Constitution_avg=avg(Constitution)
#
# print('\nC1课程平均成绩：%.2f'% C1_avg)
# print('C2课程平均成绩：%.2f'% C2_avg)
# print('C3课程平均成绩：%.2f'% C3_avg)
# print('C4课程平均成绩：%.2f'% C4_avg)
# print('C5课程平均成绩：%.2f'% C5_avg)
# print('C6课程平均成绩：%.2f'% C6_avg)
# print('C7课程平均成绩：%.2f'% C7_avg)
# print('C8课程平均成绩：%.2f'% C8_avg)
# print('C9课程平均成绩：%.2f'% C9_avg)
# print('体能测试平均成绩：%.2f'% Constitution_avg)

#标准差公式
def std(df_std,df_avg):
    count=0
    result=0
    for i in range(len(df_std)):
        count+=((df_std[df_std.index[i]]-df_avg)**2)
        result=(count/(len(df_std)-1))**0.5
    return result

# #计算各门成绩的标准差
# C1_std=std(df['C1'],C1_avg)
# C2_std=std(df['C2'],C2_avg)
# C3_std=std(df['C3'],C3_avg)
# C4_std=std(df['C4'],C4_avg)
# C5_std=std(df['C5'],C5_avg)
# C6_std=std(df['C6'],C6_avg)
# C7_std=std(df['C7'],C7_avg)
# C8_std=std(df['C8'],C8_avg)
# C9_std=std(df['C9'],C9_avg)
# Constitution_std=std(Constitution,Constitution_avg)
#
# print('\nC1课程标准差：%.2f'% C1_std)
# print('C2课程成绩标准差：%.2f'% C2_std)
# print('C3课程成绩标准差：%.2f'% C3_std)
# print('C4课程成绩标准差：%.2f'% C4_std)
# print('C5课程成绩标准差：%.2f'% C5_std)
# print('C6课程成绩标准差：%.2f'% C6_std)
# print('C7课程成绩标准差：%.2f'% C7_std)
# print('C8课程成绩标准差：%.2f'% C8_std)
# print('C9课程成绩标准差：%.2f'% C9_std)
# print('体能测试成绩标准差：%.2f'% Constitution_std)
#
# #相关系数公式
# def correlation(df1, df2,df1_avg,df1_std,df2_avg,df2_std):
#     result=0
#     for i in range(len(df1)):
#         a=(df1[df1.index[i]] - df1_avg) / df1_std
#         b=(df2[df2.index[i]] - df2_avg) / df2_std
#         result += a*b
#     return result
#
#
# print('\nC1课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C1'], Constitution,C1_avg,C1_std,Constitution_avg,Constitution_std))
# print('C2课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C2'], Constitution,C2_avg,C2_std,Constitution_avg,Constitution_std))
# print('C3课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C3'], Constitution,C3_avg,C3_std,Constitution_avg,Constitution_std))
# print('C4课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C4'], Constitution,C4_avg,C4_std,Constitution_avg,Constitution_std))
# print('C5课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C5'], Constitution,C5_avg,C5_std,Constitution_avg,Constitution_std))
# print('C6课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C6'], Constitution,C6_avg,C6_std,Constitution_avg,Constitution_std))
# print('C7课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C7'], Constitution,C7_avg,C7_std,Constitution_avg,Constitution_std))
# print('C8课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C8'], Constitution,C8_avg,C8_std,Constitution_avg,Constitution_std))
# print('C9课程成绩与体能测试成绩的相关性为：%.2f'% correlation(df['C9'], Constitution,C9_avg,C9_std,Constitution_avg,Constitution_std))
