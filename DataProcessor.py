import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime,date
from datetime import timedelta
import pandasql as pdsql

from pandasql import sqldf


data=pd.read_csv('C:/Analytics/dataset/data_science_challenge_samp_18.csv')
data2=pd.read_csv('C:/Analytics/dataset/data_science_challenge_samp_18.csv')

data2['order_date']=pd.to_datetime(data2['order_date'])
data2['yyyy_w']=data2['order_date'].apply(lambda x: str(x.isocalendar()[0])+'-'+str(x.isocalendar()[1]))

q1 ="select cust_id,yyyy_w,sum(units_purchased)as total_purchased,sum(total_spend)as total_spent from data2 group by cust_id,yyyy_w;"
f1=pdsql.sqldf(q1)
#print(f1)
cats = ['a', 'b', 'c']
df4 = pd.DataFrame({'cat': ['a', 'b', 'a']})

dummies = pd.get_dummies(df4, prefix='', prefix_sep='')
dummies = dummies.T.reindex(cats).T.fillna(0)

lane=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
print("")
dummies=pd.get_dummies(data['lane_number'],prefix='', prefix_sep='')
dummies = dummies.T.reindex(lane).T.fillna(0)
#print(dummies)

#print(data)
#print(type(data))
#print(type(dummies))
original=pd.concat([data,dummies], axis=1)

#print(original)
data['order_date']=pd.to_datetime(data['order_date'])
data['yyyy_w']=data['order_date'].apply(lambda x: str(x.isocalendar()[0])+'-'+str(x.isocalendar()[1]))

#print(data)
#data['visit']


start_date = min(data['order_date'])
end_date   = max(data['order_date'])
#print(str(start_date))
#print(str(datetime.now()+timedelta(4)))
dates= [ start_date + timedelta(n) for n in range(int ((end_date - start_date).days))]
#print(int ((end_date - start_date).days))
#print(type(dates))
#dates=pd.to_datetime(data['dates'])
#dates=dates.to_frame()
df3=pd.DataFrame(np.array(dates))
df3.columns=['order_date']
#print(df3)
#print(type(df3))
df4=data.head(10)
q2 ="select distinct df3.order_date, df4.cust_id from df3, df4;"

f2=pdsql.sqldf(q2)
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
print(f2.loc(0,'order_date'))
f2['order_date']=pd.to_datetime(f2['order_date'])
#print(f2)
#print(df4)
#print(df4.cust_id)
#q3= "select * from df4 left join f2 on df4.order_date=f2.order_date where df4.cust_id=f2.cust_id; "
#f3=pdsql.sqldf(q3)
#print(f3)
f3=f2.head(10)
print (f3)

print('****************************************')

#df4['order_date']=pd.to_datetime(df4['order_date'])
print(df4)
print('***************************************')
result=pd.merge(f3,df4, on=['cust_id','order_date'], how='left')
print (result.head(500));
result.to_csv('C:/Analytics/dataset/test2.csv', sep=',' )
#for index, row in df.iterrows():
left1 = pd.DataFrame({'key': ['K0', 'K1', 'K54', 'K3','k4','K0'],
                       'A': ['A0', 'A1', 'A2', 'A3','A4','A1'],
                      'B': ['B0', 'B1', 'B2', 'B3','B4','B5']})
 

print(left1)
right1 = pd.DataFrame({'key': ['K0', 'K0', 'K21', 'K3'],
                       'A': ['A0', 'A1', 'A2', 'A3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})

print(right1)

result=pd.merge(left1,right1, on=['key','A'], how='left')
print(result)




      

