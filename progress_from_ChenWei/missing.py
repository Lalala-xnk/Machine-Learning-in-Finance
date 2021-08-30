#数据探索

#数据预处理
import pandas as pd
df = pd.read_csv(r'C:\Users\17517\Desktop\毕业论文\LoanStats_2017Q1.csv', low_memory=False)
#原始变量145个

#（1）数据泄露变量卸载
#贷后变量删除
leak_feas=['initial_list_status','out_prncp','out_prncp_inv','total_pymnt',
           'total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee',
           'recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',
           'next_pymnt_d','last_credit_pull_d']
df.drop(leak_feas,axis=1,inplace=True)

#（2）缺失值处理
#严重缺失值处理
#缺失90%以上的变量删除
miss90=[i for i in df.columns if (df[i].isnull().sum()*1.0/df.shape[0]) > 0.9 ]
df=df.drop(miss90,axis=1)
#缺失40%以上的变量分析
miss40=[i for i in df.columns if (df[i].isnull().sum()*1.0/df.shape[0]) > 0.4]
import xlsxwriter
workbook=xlsxwriter.Workbook(r'C:\Users\17517\Desktop\毕业论文\miss.xlsx')
worksheet40=workbook.add_worksheet('miss40')
m=0
for i in miss40:
    mis_rate=df[i].isnull().sum()*1.0/df.shape[0]
    worksheet40.write(m,0,i)
    worksheet40.write(m,1,mis_rate)
    m=m+1
#    print(i,'\t',mis_rate)
#对变量进行分析，确定缺失值处理方式
#mths_since_last_delinq 自最近一次拖欠的月数
#mths_since_last_record 自上次公开黑记录以来的月数
#mths_since_last_major_derog 自更糟糕的评级以来的月数（删除）
#mths_since_recent_bc_blq 自最近银行卡出现拖欠的月数
#mths_since_recent_revol_deling 自最近一次循环拖欠的月数（删除）
df.drop('mths_since_last_major_derog',axis=1,inplace=True)
df.drop('mths_since_recent_revol_delinq',axis=1,inplace=True)
#其余缺失变量
miss00=[i for i in df.columns if df[i].isnull().sum() != 0]
worksheet00=workbook.add_worksheet('miss00')
n=0
for i in miss00:
    mis_rate=(df[i].isnull().sum())*1.0/df.shape[0]
    worksheet00.write(n, 0, i)
    worksheet00.write(n, 1, mis_rate)
    n = n + 1
#    print(i,'\t',mis_rate)
workbook.close()
#删除缺失小于1%的变量中有缺失值的行
miss_del=[i for i in df.columns if (df[i].isnull().sum()*1.0/df.shape[0]) < 0.01]
df=df.dropna(axis=0,how='any',subset=miss_del)
#用众数填充部分变量
miss_fill=[i for i in df.columns if 0.0 < (df[i].isnull().sum()*1.0/df.shape[0]) < 0.4]
from scipy.stats import mode
for i in miss_fill:
    df[i][df[i].isnull()]=mode(df[i][df[i].notnull()])[0][0]

#处理同一性数据
from scipy.stats import mode
equi_fea=[]
for i in df.columns:
    try:
        mode_value=mode(df[i])[0][0]
        mode_rate=mode(df[i])[1][0]*1.0 / df.shape[0]
        if mode_rate > 0.9:
            equi_fea.append([i,mode_value,mode_rate])
    except:
        pass
e=pd.DataFrame(equi_fea,columns=['col_name','mode_value','mode_rate'])
e.sort_values(by='mode_rate')
equi_drop=list(e.col_name.values)
df.drop(equi_drop,axis=1,inplace=True)

#特征格式变换
df.term=df.term.str.replace('months','').astype('float')
df.int_rate=df.int_rate.str.replace('%','').astype('float')
df.revol_util=df.revol_util.str.replace('%','').astype('float')
df.drop(['earliest_cr_line','issue_d'],axis=1)

df.info()
