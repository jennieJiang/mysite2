import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
pd.set_option('display.max_columns', None)  #输出所有列

#1) import datasets
df_seq=pd.read_csv('pdb_data_seq.csv')
df_char=pd.read_csv('pdb_data_no_dups.csv')
print('Datasets have been loaded...')

#2) filter and process datasets
#filter for only protein  macromoleculeType
protein_chair=df_char[df_char.macromoleculeType=='Protein']
protein_seq=df_seq[df_seq.macromoleculeType=='Protein']
#print(protein_seq.head())
#print('--------------------')

#select only necessary variables to join
protein_chair=protein_chair[['structureId','classification']]
protein_seq=protein_seq[['structureId','sequence']]
#print(protein_seq.head())
#print(protein_chair.head())

#join two datasets om structureId
model_f=protein_chair.set_index('structureId').join(protein_seq.set_index('structureId'))  #join默认以index进行合并
print(model_f.head())
#print(model_f.shape[0])
print('--------------------')
print('%d is the number of rows in the joined datasets' % model_f.shape[0])    #join后总共有%d个蛋白

#缺失值处理 check NA counts
print(model_f.isnull().sum())   #返回数据每一列缺失值个数

#缺失值过滤 drop rows with missing values
model_f=model_f.dropna()   #只要这一行含有缺失>=1个缺失值则删除这一行
print('%d is the number of proteins that have a classification and sequence' % model_f.shape[0])

#look at classification type counts
counts=model_f.classification.value_counts()
print(counts)

#plot counts
'''plt.figure()
sns.distplot(counts,hist=False,color='purple')
plt.title('count distribution for family types')
plt.ylabel('% of records')
plt.show()'''

#get classification types where counts are over 1000
#print(counts[(counts > 1000)])       #显示counts超过1000的数据
print(counts[(counts > 1000)].index)   #超过1000的数据对应的classification名称
types=np.asarray(counts[(counts > 1000)].index)  #数据类型转换为ndarray即numpy数组类型
#print(types)

#filter dataset's records for classification types > 1000
print(model_f.classification.isin(types))  #isin()表示types 对应的分类名称是否在model_f的classification列，如果在返回对应structureId为True,否则返回False
data=model_f[model_f.classification.isin(types)]    #选取True对应的index,即data选取model_f[10GS]的数据（上一步返回10GS：True），此时10GS其实是dataFrame类型的model_f的index
print(data)
print('%d is the number of records in the final filtered dataset' % data.shape[0])    #有%d个分类，其数量超过1000个

#3）Train Test Split
#split data
X_train,X_test,y_train,y_test=train_test_split(data['sequence'],data['classification'],test_size=0.2,random_state=1)

#creat a count vectorizer to gather the unique elements in sequence----特征提取
vect=CountVectorizer(analyzer='char_wb',ngram_range=(4,4))   #char_wb表示以4个字母为一个窗口进行祠内扫描

#Fit and Transform CountVectorizer----特征提取
vect.fit(X_train)
X_train_df=vect.transform(X_train)    #上述两步等同于vect.fit_transform(X_train)，即提取词频
X_test_df=vect.transform(X_test)

#print a few of the features
print(vect.get_feature_names()[-20:])
print('X_train_df',X_train_df)

#4) Machine Learning Models
#make a prediction dictionary to store accuracys
prediction=dict()      #效果等同于prediction={}

#Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_df,y_train)
NB_pred=model.predict(X_test_df)
prediction['MultinomialNB']=accuracy_score(y_test,NB_pred)
print('NB_accuracy',prediction['MultinomialNB'])

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier()
model.fit(X_train_df,y_train)
Adaboost_pred=model.predict(X_test_df)
prediction['AdaBoostClassifier']=accuracy_score(y_test,Adaboost_pred)
print('AdaBoostClassifier_Accuracy',prediction['AdaBoostClassifier'])

#5) plot confusion matrix for NB
#plot confusion matrix
conf_mat=confusion_matrix(y_test,NB_pred,labels=types)   #labels如果不设置，默认是y_test,NB_pred两者中出现至少一次的标签名

#Normalize confusion matrix
conf_mat=conf_mat.astype('float')/conf_mat.sum(axis=1)[:,np.newaxis]

#plot heat map
fig,ax=plt.subplots()
fig.set_size_inches(13,8)
sns.heatmap(conf_mat)
plt.show()

#Print F1 score metrics
print(classification_report(y_test, NB_pred, target_names = types))







