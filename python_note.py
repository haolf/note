#运算符
'+' #加号
'-' #减号
'*' #乘号
'/' #除号
'**'#次幂
'//'#整除
'%' #余数
'>' #大于
'<' #小于
'=' #赋值
'=='#等于
'!='#不等于
'>='#大于等于
'<='#小于等于
'\''#反斜杠转义引号

file()              #文件
tuple()             #元组
dict()              #字典
list()              #列表
bool()              #布尔值
int()               #转换为整数
float()             #转换为浮点数
str()               #转换为字符串
type()              #返回数据类型
len()               #返回字符长度或元素个数
max()               #返回最大的元素
min()               #返回最小的元素
sum()               #求和,axis设置轴,0为列,1为行
sorted()            #按从小到大的顺序返回列表的副本
round(float,int)    #返回一个浮点数的四舍五入近似值,取小数点后int位,默认为整数
range()             #一个范围
open('file','r')    #以只读方式打开文件,'w'可写入文件
enumerate(s,start=0)#将一个可遍历的对象(如列表、元组或字符串)组合为一个索引序列

# open示例
f = open('/my_path/my_file.txt','r')
f.read()         #读取指定参数的字符,默认全部
f.close          #关闭释放文件
with open('/my_path/my_file.txt','r') as f:
    file_data = f.read()

# enumerate示例
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons, start=1))       #小标从1开始,默认为0.
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

#普通的 for 循环                     #enumerate循环
>>>i = 0                           |
>>> seq = ['one', 'two', 'three']  | >>>seq = ['one', 'two', 'three']
>>> for element in seq:            | >>> for i, element in enumerate(seq):
...     print i, seq[i]            | ...     print i, seq[i]
...     i +=1                      | ...
...                                |
0 one                              | 0 one
1 two                              | 1 two
2 three                            | 2 three

str.title()       #返回首字母大写的字符串
str.upper()       #返回字符转换为大写字母
str.lower()       #返回字符转换为小写字母
str.islower()     #说明该字符串对象中的字母是否都是小写字母
str.count(str)    #返回字符串中子字符串 指定值 出现的次数
str.format(x)     #将值插入到模板字符串中的'{}'来构建字符串
str.split(str)    #将给定字符串以指定符号分割为多份,返回为列表(expand=True扩张)
str.join(list)    #用字符串来连接列表元素
str.strip(str)    #去除字符串首尾指定字符,默认为空格
list.sort()       #从小到大排序列表
list.append(x)    #在列表末尾添加一个x元素
file.readine()    #读取文件下一行
bool.all()        #确认所有布尔值
def xxxx(xx=0):   #函数默认参数为0
    '''
    xxxx
    '''           #说明
    if xx:
        pass
    elif xx:
        pass
    else:
        break     #停止
    for x in xx:  #for循环
        pass
    while xx:     #while循环
        break

    return x #返回值

print() #打印
#匿名函数,冒号前是参数,可以有多个,用逗号隔开,冒号右边的返回值。
lambda x : x**2     #输入参数x,返回x**2
g=lambda x : x**2   #命名
g(x) == x**2

#数据集处理
import pandas as pd

#导入数据
pd.read_csv(filename)                #从CSV文件导入数据
pd.read_table(filename)              #从限定分隔符的文本文件导入数据
pd.read_excel(filename)              #从Excel文件导入数据
pd.read_sql(query, connection_object)#从SQL表/库导入数据
pd.read_json(json_string)            #从JSON格式的字符串导入数据
pd.read_html(url)                    #解析URL,字符串或HTML文件,抽取其中的tables表格
pd.read_clipboard()                  #从你的粘贴板获取内容，并传给read_table()
pd.DataFrame(dict)                   #从字典对象导入数据，Key是列名，Value是数据
#导出数据
df.to_csv('xxx_edited.csv',index=False) #保存为csv文件,index不保存索引
df.to_excel(filename)                   #导出数据到Excel文件
df.to_sql(table_name, connection_object)#导出数据到SQL表
df.to_json(filename)                    #以Json格式导出数据到文本文件


# 读取CSV文件
df = pd.read_csv('xxx.csv')                    #从CSV文件导入数据
df = pd.read_csv('xxx.csv',sep=';')            #用';'替换','作为分隔符
df = pd.read_csv('xxx.csv',index_col=['a'])    #设置指定列或多个列作为索引
df = pd.read_csv('xxx.csv',header=None)        #设置指定行为标签,None时无标签
df = pd.read_csv('xxx.csv',header=0, names=[]) #另设指定标签

pd.to_datetime(df['a'])  #将a列转换数据类型为datetime
pd.to_numeric(df['a'],errors='raise')
                         #将a列转换数据类型为数值,
                         #errors='coerce'转换无效值为NaN
                         #errors='ignore'忽略无效值
df.astype(int)#强制转换数据类型为int
df.head()     #显示文件前指定行,默认为5
df.tail()     #显示文件后指定行,默认为5
df.shape      #返回列表的行数与列数
df.index      #查看索引
df.count()    #返回每列的元素个数
df.dtypes     #返回每列数据的数据类型
df.columns    #返回所有标签,以列表的形式
df.info()     #显示数据框的简明摘要.包括每列非空值的数量
df.describe() #查看数据值列的汇总统计
df.quantile() #返回给定的分位数,默认为0.5(中位数50%)
df.mean()     #返回每一列的均值
df.max()      #返回每一列的最大值
df.min()      #返回每一列的最小值
df.median()   #返回每一列的中位数
df.std()      #返回每一列的标准差
df.sum()      #返回每一列的和
df.argmax()   #返回最大值的索引键
df.argmin()   #返回最小值的索引键
df.copy()     #返回一个df的副本
http://www.cnblogs.com/lemonbit/p/7270045.html #将一列中的文本拆分为多行的详细文档
df. loc[:,:]  #选择指定的行与列,以字符串索引
df.iloc[:,:]  #选择指定的行与列,以数字索引
df.iloc[:,np.r_[0,2,4:7]]           #选择分开的行与列,需导入np
df['a'].unique()                    #查看Series对象的唯一值
df['a'].nunique()                   #查看series对象有多少个唯一值
df['a'].value_counts()              #查看Series对象的唯一值和计数
df['a'].isin([x])                   #返回一个布尔值,查询列表中值是否为指定值x
df['a'].index                       #返回
df.dorp(labels = None，axis = 0(索引)1(标签),|index = None，columns = None)
df.drop(['B', 'C'], axis=1)         #删除列表中指定的'B'列和'C'列
df.drop(columns=['B','C'])          #删除列表中指定的'B'列和'C'列
df.drop([0, 1])                     #删除列表中的0行和1行
df.duplicated().sum()               #检查数据中的重复
df.drop_duplicates(inplace=True)    #删除列表中的重复并原地修改
df.isnull()                         #返回布尔值,查看列表元素是否缺失
df[df.isnull().values==True]        #返回所有存在缺失值的行
df.dropna(inplace=True)             #返回所有无缺失值的行
df['a'].fillna(num,inplace=True)    #用num填充缺失值,inplace参数原地修改
df.sort_values(by=['a'])            #升序排列指定列a,参数ascending=False降序排列
a=pd.DataFrame([[1,2,3]],columns=['a','b','c'])
df.apppend(a,ignore_index=True)     #将列表a合并到df末尾,ignore_index重新配置索引
df.rename(index=str, columns={'old':'new'},inplace=True)
                                    #替换列表中某列单个的标签
                                    #old为需要被替换的旧标签,new为新标签
df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
                                    #将标签中的空格替换为下划线
df['a'].str.extract('(\d+)').astype(int)
                                    #提取字符串中的整数,并将数据类型转换为int
df[df['a'].str.contains('/')]       #返回a列中所有元素的字符串里里包含'/'的行
df.groupby('a')['b'].module()       #以a列中每个唯一值为索引,返回其他列的统计数据
                                    #as_index=False时显示数字索引,默认True
pd.cut(x,bin,labels=None)           #对应x列中数据的值,返回一个新列,需赋值以保存
                                    #x:    一维数组(例如df['a'])
                                    #bin:  指定范围(例如[1,2,3])
                                    #label:指定列表,为每个范围命名,默认为整数指示符
df[df['a'] == A]                    #返回a列中所有值为A的框架列表,A的类型需相同
df[df['a'].isin([A])]               #返回a列中指定值为A的框架列表
df.query('a == A')                  #返回a列中指定值为A的框架列表,引用变量时前缀@
df.query('a in ["A", "B"]')         #返回a列中指定值为A或B的框架列表
df.apply(f,axis = 0)                #f为自定函数|def f(x)|lambda x:x
                                    #axis默认0将函数应用于每列,1将函数应用于每行
pd.merge(df_left,df_right,how='inner',on=None,left_on=None,right_on=None)
                                    #以某列为媒介合并连接数据集,需赋值以储存
                                    #how:inner交集,outer并集,left或right表取某边
                                    #on:连接的列名;left_on\right_on连接不同尘列名
df1.join(df2)                       #连接数据集,需赋值以储存

%matplotlib inline #允许网页中绘图
df.hist(figsize=(8,8));            #绘制直方图,figsize设置大小,';'隐藏不需要的内容
df['a'].hist();                                #绘制a列的直方图
df['a'].plot(kind='hist');                     #绘制a列的直方图
df['a'].value_counts().plot(kind='bar');       #绘制a列唯一值频数的柱形图
df['a'].value_counts().plot(kind='pie');       #绘制a列唯一值频数的饼状图
df['a'].plot(kind='box');                      #绘制a列的箱型图
df.plot(x='a',y='b',kind='scatter');           #绘制变量a与b的散点图
pd.plotting.scatter_matrix(df,figsize=(15,15));#绘制所有变量的散点图
pd.get_dummies(df['a'])                        #把a列唯一值转换虚拟变量 是为1否为0


import numpy as np
array.size                       #获得所有数据的个数
np.exp(x)                        #获得e的x次方幂
np.sqrt(5)                       #获得'5'的开方
n.repeat(a,repeat,axis=None)     #重复指定数组(a输入数组,repeat设置重复,axis设置轴
df.iloc[:,np.r_[0,2,4:7]]        #选择分开的行与列,需导入np
#注:以下函数默认replace=True,即默认放回样本
np.random.randint(2,size=(100,2))#[low,high),返回随机整数
                                 # 0,2 :在0和1之中随机选取一个数,0已省略
                                 #重复100个2次,生成1列表包含100个列表包含2个元素
np.random.choice([0, 1],size=100,p=[0.8,0.2])
                                 #[0,1]:选择范围在0和1之中随机选取一个数
                                 #size:重复100次,生成列表
                                 #p:设置概率,默认为相同
                                 #replace=False 时,不放回抽样
np.random.binomial(n, p, size)   #测试n次概率事件,概率为p,统计size轮
np.var(a)                        #求数组a的方差
np.std(a)                        #求数组a的标准差
np.percentile(a,50)              #求数组a的50%分位数
df.sample(100,replace=False)     #随机获取列表df的一百行个样本,默认不放回抽样
np.random.normal(loc,scale,size) #模拟正太分布
                                 #loc为分布均值
                                 #scale为标准差
                                 #size为重复次数,默认为None,只输出一个值

import matplotlib.pyplot as plt
#figsize设置统计图大小,alpha设置统计图透明度
plt.subplots(figsize=(10,5))               #设置统计图大小
plt.bar([1,2,3], [224,620,425],color='br') #绘制一个柱形图,[1,2,3]为横坐标
plt.xticks([1,2,3], ['a','b','c']);        #为x轴标签重命名
plt.bar([1,2,3], [224,620,425], tick_label=['a', 'b', 'c']);#合并上面两个函数
plt.title('Some Title')                    #设置总标题
plt.xlabel('Some X Label',figsize=10)      #设置横坐标标题
plt.ylabel('Some Y Label',figsize=5);      #设置纵坐标标题
plt.hist(a)                                #绘制数据a的直方图
plt.axvline(x=1,color='r',linewidh=2)      #设置分割线,值为1,颜色为'r',宽为2.


import seaborn as sb
sb.pairplot(df[['a','b','c']])             #绘制统计图观察a,b,c列之间的关系


import statsmodels.api as sm

#拟合基础流程
df['intercept'] = 1                        #设置截距列
lm = sm.OLS(df['y'],df[['intercept','x']]) #设置数据集中的自变量和因变量(最小二乘法)
results = lm.fit()                         #拟合模型并储存
results.summary()                          #查看摘要
#转换虚拟变量
df[['A','B','C']] = pd.get_dummies(df['a'])#转换a列虚拟变量设置入新列ABC
lm=sm.Logit(df['y'],df[['intercept','x']]) #逻辑回归中不使用最小二乘法
results = lm.fit()                         #拟合模型并储存
results.summary2()                         #查看摘要
#计算vif值
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

y,x=dmatrices('price ~ area + bedrooms + bathrooms',df,return_type='dataframe')

vif=pd.DataFrame()
vif['VIF Factor']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
vif['features']=x.columns


import sklearn.preprocessing as p
p.scale(df['a'])                           #获取a列的缩放特征(减去均值并除以标准差)
