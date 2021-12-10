import pandas as pd
events = pd.read_csv('Enron_events.txt', sep="	")
positions=pd.read_csv('Enron_positions.txt', sep='	')
positions.columns=['id','email','position']
from datetime import datetime
#pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

date=[]
for i in events['time']:
    day=datetime.fromtimestamp(956834340 + i).strftime("%d-%m-%Y")
    day=datetime.strptime(day,"%d-%m-%Y")
    date.append(day)

events['date']=date

###merge data sets to give more information about senders
new_df= pd.merge(
        events,
        positions,
        how='left',
        left_on='exped',
        right_on='id'
    )




new_df=new_df.rename(columns={'id':'sender_id', 'email':'sender_email','position':'sender_position'})
new_df=new_df.drop('exped',axis=1)
enron= pd.merge(
        new_df,
        positions,
        how='left',
        left_on='destin',
        right_on='id'
    )
enron=enron.rename(columns={'id':'destin_id', 'email':'destin_email','position':'destin_position'})

enron=enron.drop('destin',axis=1)

enron_sender_group=enron.groupby(['sender_id','date'])['destin_id'].count().reset_index(name="count_sent")
enron_rec_group=enron.groupby(['destin_id','date'])['sender_id'].count().reset_index(name="count_rec")

#find count of emails sent and received per day per employee
profiles=pd.merge(
    enron_rec_group,
    enron_sender_group,
    left_on=['destin_id','date'],
    right_on=['sender_id','date']
)
profiles=profiles.rename(columns={'sender_id':'id', 'sender_email':'email','sender_position':'position'})
daily_count_per_profile=profiles[['date','id','count_sent','count_rec']]


#####
###find count of emails sent and received in total (dropping day variable from grouping)

enron_sender_group_agg=enron.groupby(['sender_id','sender_position','sender_email'])['destin_id'].count().reset_index(name="count_sent")
enron_rec_group_agg=enron.groupby(['destin_id'])['sender_id'].count().reset_index(name="count_rec")

profiles_agg=pd.merge(
    enron_rec_group_agg,
    enron_sender_group_agg,
    left_on='destin_id',
    right_on='sender_id'
)
profiles_agg=profiles_agg.rename(columns={'sender_id':'id', 'sender_email':'email','sender_position':'position'})
profiles_agg=profiles_agg[['id','email','position','count_sent','count_rec']]




###find average daily emails sent and recieved
enron_daily_emails_sent = events.groupby([events['date'].dt.date,events['exped']])['destin'].count().reset_index().rename(columns={'destin':'daily_count_sent','exped' : 'employee_id'})
enron_daily_emails_received = events.groupby([events['date'].dt.date,events['destin']])['exped'].count().reset_index().rename(columns={'destin':'employee_id','exped' : 'daily_count_received'})
employee_email_stats_daily= pd.merge(
        enron_daily_emails_received,
        enron_daily_emails_sent,
        how='outer',
        on=['employee_id','date']
    )

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

