from Enron_Project_Data import positions, employee_email_stats_average
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn import cluster
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from matplotlib import pyplot as plt
pd.set_option("display.max_columns", None)

#scale employee average sent/recieved data to prepare for k-means clustering
mms = StandardScaler()
mms.fit(employee_email_stats_average)
data_transformed = mms.transform(employee_email_stats_average)
Sum_of_squared_distances = []
print(data_transformed)
# find ideal number of clusters using silhouette scores,elbow method
K = range(2,15)
silhouette_scores = []
CS_scores = []
for k in K:
    km = KMeans(n_clusters=k, init='k-means++', random_state=1)
    cluster_found = km.fit_predict(data_transformed)
    silhouette_scores.append(metrics.silhouette_score(data_transformed, km.labels_))
    Sum_of_squared_distances.append(km.inertia_)
    CS_scores.append(metrics.calinski_harabasz_score(data_transformed, km.labels_))

plt.plot(np.arange(2,15), silhouette_scores)
plt.xlabel('k')
plt.title('Find Optimal k using Silhouette')
plt.savefig('silhouette.png')
plt.show()

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.title('Find Optimal k using elbow')
plt.savefig('elbow_score.png')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=1)
y = kmeans.fit_predict(data_transformed)

employee_email_stats_average['Cluster'] = y
LABEL_COLOR_MAP={0 : 'r',
                 1:'b',
                 2:'y',
                 3: 'g',
                 4: 'm'}
#
label_color = [LABEL_COLOR_MAP[l] for l in employee_email_stats_average['Cluster']]
ax=plt.scatter(employee_email_stats_average['daily_count_sent'] , employee_email_stats_average['daily_count_received'],c=label_color)
plt.title('Clusters based on average emails sent and received daily')
plt.xlabel('daily_count_sent')
plt.ylabel('daily_count_received')

plt.savefig('cluster.png')

plt.show()





employee_email_stats_average= pd.merge(
        employee_email_stats_average,
        positions,
        how='left',
        left_on='employee_id',
        right_on='id'
    )

#
group1=employee_email_stats_average.where(employee_email_stats_average['Cluster']==0)
group2=employee_email_stats_average.where(employee_email_stats_average['Cluster']==1)
group3=employee_email_stats_average.where(employee_email_stats_average['Cluster']==2)
group4=employee_email_stats_average.where(employee_email_stats_average['Cluster']==3)
group5=employee_email_stats_average.where(employee_email_stats_average['Cluster']==4)
print(group1)

count_positions_group1=group1.groupby(group1['position'])['position'].count().reset_index(name="count")
count_positions_group2=group2.groupby(group2['position'])['position'].count().reset_index(name="count")
count_positions_group3=group3.groupby(group3['position'])['position'].count().reset_index(name="count")
count_positions_group4=group4.groupby(group4['position'])['position'].count().reset_index(name="count")
count_positions_group5=group5.groupby(group5['position'])['position'].count().reset_index(name="count")
print(count_positions_group1)


count_cluster_by_position=employee_email_stats_average.groupby(['Cluster','position'])['id'].count()
count_cluster_by_position=pd.DataFrame(count_cluster_by_position)
html_pos=count_cluster_by_position.to_html()
pd.set_option("display.max_rows", None)

employee_cluster=employee_email_stats_average[['Cluster','id','position']]
html_emp=employee_cluster.to_html()
# import numpy as np


N=5
width = 0.30       # the width of the bars

ind = np.arange(N)  # the x locations for the groups

fig = plt.figure(figsize = (10, 8), dpi = 100) #dpi=100 instead of dpi=256
ax = fig.add_axes([0.1,0.2,0.8,0.7])
five=ax.bar(count_positions_group5['position'],count_positions_group5['count'],color='m',zorder=4)
four=ax.bar(count_positions_group4['position'],count_positions_group4['count'],color='g',zorder=3)
three=ax.bar(count_positions_group3['position'],count_positions_group3['count'],color='y',zorder=5)
two=ax.bar(count_positions_group2['position'],count_positions_group2['count'],color='b',zorder=1)
one=ax.bar(count_positions_group1['position'],count_positions_group1['count'],color='r',zorder=2)

ax.legend(labels=['4','3','2','1','0'])
plt.xticks(rotation=90)
plt.minorticks_on()
plt.title('Clusters based on position')
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.005*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(five)
autolabel(four)
autolabel(three)
autolabel(two)
autolabel(one)

plt.savefig('positions.png')

plt.show()




