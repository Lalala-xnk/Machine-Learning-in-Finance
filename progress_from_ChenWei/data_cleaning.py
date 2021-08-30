#课题分析

#数据获取
import pandas as pd
df = pd.read_csv(r'C:\Users\17517\Desktop\毕业论文\LoanStats_2017Q1.csv', low_memory=False)
#df.info()
#数据探索

#数据预处理
#1.缺失值处理
#（1）删除确实比例90%以上的变量
mis_90=[i for i in df.columns if (df[i].isnull().sum()*1.0/df.shape[0]) > 0.9]
df.drop(mis_90,axis=1,inplace=True)
#df.info()
#df.to_csv(r'C:\Users\17517\Desktop\毕业论文\result.csv')
#（2）删除贷后变量
fea_drop=['initial_list_status','out_prncp','out_prncp_inv','total_pymnt',
          'total_pymnt_inv','total_rec_prncp','total_rec_int',
          'total_rec_late_fee','recoveries','collection_recovery_fee',
          'last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d']
df.drop(fea_drop,axis=1,inplace=True)
#df.info()
#（3）删除含缺失10%以下变量的记录
mis_10=[i for i in df.columns if (df[i].isnull().sum()*1.0/df.shape[0]) < 0.1]
df=df.dropna(axis=0,how='any',subset=mis_10)
#df.info()
#（4）展示缺失变量及缺失比例，确定处理策略
mis=[i for i in df.columns if df[i].isnull().sum() != 0]
for i in mis:
    mis_rate=(df[i].isnull().sum())*1.0/df.shape[0]
    #print(i,' ',mis_rate)
values={'mths_since_last_delinq':999,
        'mths_since_last_record':999,
        'mths_since_last_major_derog':999,
        'il_util':999,
        'mths_since_recent_bc_dlq':999,
        'mths_since_recent_inq':999,
        'mths_since_recent_revol_delinq':999}
df.fillna(value=values,inplace=True)

#2.同值性特征识别和处理
from scipy.stats import mode
equi=[]
for i in df.columns:
    try:
        mode_value=mode(df[i])[0][0]
        mode_rate=mode(df[i])[1][0]*1.0 / df.shape[0]
        if mode_rate > 0.9:
            equi.append([i,mode_value,mode_rate])
    except:
        pass
e=pd.DataFrame(equi,columns=['col_name','mode_value','mode_rate'])
e.sort_values(by='mode_rate')
print(e)
equi_fea=list(e.col_name.values)
df.drop(equi_fea,axis=1,inplace=True)

#3.格式变换
df.term=df.term.str.replace('months','').astype('float')
df.int_rate=df.int_rate.str.replace('%','').astype('float')
df.revol_util=df.revol_util.str.replace('%','').astype('float')
df.drop(['earliest_cr_line','issue_d'],axis=1,inplace=True)

df.info()

#4.文本特征处理
#工作机构的分类
df.drop('emp_title',axis=1,inplace=True)
#借款人州地址分类
df.drop('addr_state',axis=1,inplace=True)

#6.删除无意义变量
df.drop(['sub_grade','zip_code'],axis=1,inplace=True)

#7.特征编码
df.grade.replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6},inplace=True)
df.emp_length.replace({"< 1 year":0,"1 year":1,"2 years":2,"3 years":3,
                          "4 years":4,"5 years":5,"6 years":6,"7 years":7,
                          "8 years":8,"9 years":9,"10+ years":10,'n/a':0},inplace=True)
df.home_ownership.replace({"MORTGAGE":0,"NONE":1,"OWN":2,"RENT":3,"ANY":4},inplace=True)
df.verification_status.replace({"Verified":0,"Source Verified":1,"Not Verified":2},inplace=True)
df.purpose.replace({"car":0,"credit_card":1,"debt_consolidation":2,"home_improvement":3,
                       "house":4,"major_purchase":5,"medical":6,"moving":7,"renewable_energy":8,
                       "small_business":9,"vacation":10,"other":11},inplace=True)
df.title.replace({"Business":0,"Car financing":1,"Credit card refinancing":2,
                     "Debt consolidation":3,"Green loan":4,"Home buying":5,
                     "Home improvement":6,"Major purchase":7,"Medical expenses":8,
                     "Moving and relocation":9,"Vacation":10,"Other":11},inplace=True)
df.term.replace({36.0:0,60.0:1},inplace=True)
df.loan_status.replace({"Fully Paid":0,"Charged Off":1,"Current":1,"Default":1,
                        "In Grace Period":0,"Late (16-30 days)":0,"Late (31-120 days)":1},
                       inplace=True)
df.info()

df.to_csv(r'C:\Users\17517\Desktop\毕业论文\data_cleaning.csv')


