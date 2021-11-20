from Enron_Project_Data import events, enron,positions
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from sklearn import cluster
from sklearn import metrics

from matplotlib import pyplot as plt
pd.set_option("display.max_columns", None)

enron_daily_emails_sent = events.groupby([events['date'].dt.date,events['exped']])['destin'].count().reset_index().rename(columns={'destin':'daily_count_sent','exped' : 'employee_id'})
enron_daily_emails_received = events.groupby([events['date'].dt.date,events['destin']])['exped'].count().reset_index().rename(columns={'destin':'employee_id','exped' : 'daily_count_received'})

# print(len(enron_daily_emails_received))

employee_email_stats_daily= pd.merge(
        enron_daily_emails_received,
        enron_daily_emails_sent,
        how='outer',
        on=['employee_id','date']
    )
# print(len(employee_email_stats_daily))

employee_email_stats_daily= pd.merge(
        employee_email_stats_daily,
        positions,
        how='left',
        left_on='employee_id',
        right_on='id'
    )
#drop n/a from employee id
employee_email_stats_daily = employee_email_stats_daily.dropna()
employee_email_stats_daily = employee_email_stats_daily.drop('id',axis=1)
employee_email_stats_daily = employee_email_stats_daily.drop('email',axis=1)
employee_email_stats_average=employee_email_stats_daily.groupby(employee_email_stats_daily['employee_id'])['daily_count_sent','daily_count_received'].mean()


#scale data
mms = MinMaxScaler()
mms.fit(employee_email_stats_average)
data_transformed = mms.transform(employee_email_stats_average)
Sum_of_squared_distances = []

# find ideal number of clusters
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

#

sil_score=[]
cs_score=[]
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

kmeans = KMeans(n_clusters=4)
y = kmeans.fit_predict(employee_email_stats_average[['daily_count_sent', 'daily_count_received']])
employee_email_stats_average['Cluster'] = y

LABEL_COLOR_MAP={0 : 'r',
                 1:'b',
                 2:'y',
                 3: 'g'}

label_color = [LABEL_COLOR_MAP[l] for l in employee_email_stats_average['Cluster']]
ax=plt.scatter(employee_email_stats_average['daily_count_sent'] , employee_email_stats_average['daily_count_received'],c=label_color)
plt.show()
#
#

employee_email_stats_average= pd.merge(
        employee_email_stats_average,
        positions,
        how='left',
        left_on='employee_id',
        right_on='id'
    )

outlier=employee_email_stats_average.where(employee_email_stats_average['Cluster']==1).dropna()
print(outlier)
# ##sort by total emails recieved
# sortedbyreccount=employee_email_stats_daily.sort_values(['daily_count_received','date'], ascending=False)
# #sort employees by total emails sent
# sortedbysentcount=employee_email_stats_daily.sort_values(['daily_count_sent','date'], ascending=False)

# employee_email_stats_daily.index = pd.to_datetime(employee_email_stats_daily["date"])
# print((employee_email_stats_daily))
# df = employee_email_stats_daily.pivot(index='date', columns='employee_id', values='daily_count_sent')
# plt.plot(df)
#
# plt.show()
#print((enron.head()))