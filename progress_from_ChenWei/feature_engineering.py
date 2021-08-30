#数据获取
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\17517\Desktop\result_cleaning.csv', low_memory=False)
#1.特征衍生

#2.筛选变量
#利用相关系数删除相关性过高的变量
def get_var_no_colinear(cutoff, df):
    corr_high = df.corr().applymap(lambda x: np.nan if x>cutoff else x).isnull()
    col_all = corr_high.columns.tolist()
    i = 0
    while i < len(col_all)-1:
        ex_index = corr_high.iloc[:,i][i+1:].index[np.where(corr_high.iloc[:,i][i+1:])].tolist()
        for var in ex_index:
            col_all.remove(var)
        corr_high = corr_high.loc[col_all, col_all]
        i += 1
    return col_all
col_cor80 = get_var_no_colinear(0.8,df)
df = df[col_cor80]
df.info()

fig,ax=plt.subplots(figsize=(20,20))
sns.heatmap(df.corr().round(2),annot=False)
plt.show()

#根据VIF消除多重共线性
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
def vif(x, thres=10.0):
    X_m = np.matrix(x)
    VIF_list = [VIF(X_m, i) for i in range(X_m.shape[1])]
    maxvif=pd.DataFrame(VIF_list,index=x.columns,columns=["vif"])
    col_save=list(maxvif[maxvif.vif<=float(thres)].index)
    col_delete=list(maxvif[maxvif.vif>float(thres)].index)
    print(maxvif)
    print('delete Variables:', col_delete)
    return x[col_save]
df=vif(df,10.0)
df.info()

#根据机器学习模型（随机森林）过滤变量
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
loan=df.copy()
d_1=loan[loan['loan_status']==1]
d_0=loan[loan['loan_status']==0]
d_1_train,d_1_test=train_test_split(d_1,test_size=.3,random_state=12)
d_0_train,d_0_test=train_test_split(d_0,test_size=.3,random_state=12)
train=pd.concat([d_1_train,d_0_train])
test=pd.concat([d_1_test,d_0_test])

train_x=train.drop(['loan_status'],axis=1)
train_y=train['loan_status']
test_x=test.drop(['loan_status'],axis=1)
test_y=test['loan_status']

rf=RandomForestClassifier(n_estimators=500,max_depth=10,random_state=1).fit(train_x,train_y)
importance=pd.DataFrame({'features':train_x.columns.values,'importance':rf.feature_importances_})
print(importance)
df=df.drop(importance[importance['importance']<0.01]['features'].values,axis=1)
df.info()

df.to_csv(r'C:\Users\17517\Desktop\毕业论文\data_selection.csv')

















