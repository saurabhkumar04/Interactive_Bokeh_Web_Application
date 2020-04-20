#!/ usr/bin/python

# Imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn
import re
import time
import warnings
import glob, os 
import functools
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from bokeh.core.properties import value
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import reset_output,output_notebook
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column,row
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from bokeh.resources import Resources
from bokeh.io.state import curstate
from bokeh.io import curdoc, output_file, save
from bokeh.util.browser import view
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import figure 
from bokeh.io import curdoc
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Div
from bokeh.io import reset_output,output_notebook
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import reset_output,output_notebook
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column,row
from datetime import date
from random import randint
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import reset_output,output_notebook
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column,row
from bokeh.plotting import figure, output_file, show
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
from math import pi
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral3
from bokeh.palettes import GnBu
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
import itertools
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column,row
from bokeh.core.properties import value
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum 
from operator import itemgetter
from bokeh.resources import Resources
from bokeh.io.state import curstate
from math import pi
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.models import LogColorMapper, LogTicker, ColorBar
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import DataTable, NumberFormatter, TableColumn, Button
from os.path import dirname, join
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import RangeSlider, Button, DataTable, TableColumn, NumberFormatter
from os.path import dirname, join

warnings.filterwarnings('ignore')

email_open = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/EmailOpen*.csv"))),ignore_index=True)
email_send = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/EmailSend*.csv"))),ignore_index=True)
email_clickthrough = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/EmailClickthrough*.csv"))),ignore_index=True)
bounceback = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/Bounceback*.csv"))),ignore_index=True)
form_submit = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/FormSubmit*.csv"))),ignore_index=True)
pageview = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/PageView*.csv"))),ignore_index=True)
subscribe = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/Subscribe*.csv"))),ignore_index=True)
unsubscribe = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/Unsubscribe*.csv"))),ignore_index=True)
web_visit = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/WebVisit*.csv"))),ignore_index=True)


email_open1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/emailOpen*.csv"))),ignore_index=True)
email_send1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/EmailSend*.csv"))),ignore_index=True)
email_clickthrough1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/EmailClickthrough*.csv"))),ignore_index=True)
bounceback1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/Bounceback*.csv"))),ignore_index=True)
form_submit1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/FormSubmit*.csv"))),ignore_index=True)
pageview1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/PageView*.csv"))),ignore_index=True)
subscribe1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/Subscribe*.csv"))),ignore_index=True)
unsubscribe1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/Unsubscribe*.csv"))),ignore_index=True)
web_visit1 = pd.concat(map(functools.partial(pd.read_csv,error_bad_lines=False), glob.glob(os.path.join(dirname(__file__), "data/new/WebVisit*.csv"))),ignore_index=True)


filter_data11 = pd.concat([email_open,email_clickthrough,email_send,bounceback,form_submit,pageview,subscribe,unsubscribe,web_visit ],ignore_index=True, sort=True)
print ("Total Records:",len(filter_data11))


email_open1 = email_open1[['Contact','Created','Email Send Type','Subject','Type','Campaign','Campaign Name','Email']]
email_click1 = email_clickthrough1[['Contact','Created','Email Send Type','Subject','Type','Campaign','Campaign Name','Email']]
email_send1 = email_send1[['Contact','Created','Email Send Type','Subject','Type','Campaign','Campaign Name','Email']]
subscribe1 = subscribe1[['Contact','Created','Type','Campaign','Campaign Name','Email']]
unsubscribe1 = unsubscribe1[['Contact','Created','Type','Campaign','Campaign Name','Email']]
bounceback1 = bounceback1[['Contact','Created','Smtp Error Code','Smtp Status Code','Type','Campaign','Campaign Name','Email']]
web_visit1 = web_visit1[['Contact','Created','Duration','First Page View Url','Number Of Pages','Type']]
form_submit1 = form_submit1[['Contact','Created','Type','Campaign','Campaign Name','Asset']]
pageview1 = pageview1[['Contact','Created','Type','Campaign','Campaign Name']]


email_open1.columns = ['contactId','created','emailSendType','subject','type','campaign','cemCampaign','email']
email_click1.columns = ['contactId','created','emailSendType','subject','type','campaign','cemCampaign','email']
email_send1.columns = ['contactId','created','emailSendType','subject','type','campaign','cemCampaign','email']
subscribe1.columns = ['contactId','created','type','campaign','cemCampaign','email']
unsubscribe1.columns = ['contactId','created','type','campaign','cemCampaign','email']
bounceback1.columns = ['contactId','created','smtpErrorCode','smtpStatusCode','type','campaign','cemCampaign','email']
web_visit1.columns = ['contactId','created','duration','firstPageViewUrl','numberOfPages','type']
form_submit1.columns = ['contactId','created','type','campaign','cemCampaign','asset']
pageview1.columns = ['contactId','created','type','campaign','cemCampaign']


filter_data12 = pd.concat([email_open1,email_click1,email_send1,bounceback1,form_submit1,pageview1,subscribe1,unsubscribe1,web_visit1 ],ignore_index=True, sort=True)
print ("Total Records:",len(filter_data12))


filter_data1= pd.concat([filter_data11,filter_data12],ignore_index=True, sort=True)
print ("Total Records:",len(filter_data1))
print ("Total Unique Contacts",filter_data1.contactId.nunique())


combined_data_mod1 = pd.DataFrame(filter_data1)
combined_data_mod1.created = pd.to_datetime(filter_data1.created)
combined_data_mod1.created = [i.date() for i in combined_data_mod1.created]

print ("Latest Date : ",max(combined_data_mod1.created))
print ("Oldest Date : ",min(combined_data_mod1.created))

filter_data1= pd.concat([filter_data11,filter_data12],ignore_index=True, sort=True)

combined_data_mod3 = pd.DataFrame(filter_data1)
combined_data_mod3.created = pd.to_datetime(filter_data1.created)
combined_data_mod3.created = [i.hour for i in combined_data_mod3.created]

filter_data2= pd.concat([filter_data11,filter_data12],ignore_index=True, sort=True)
print ("Total Records:",len(filter_data2))


def processing (filter_data1):

    combined_data = filter_data1[['contactId','created','type']]
    combined_data.sort_values(by=['created'],inplace=True)
    combined_data.reset_index(drop=True,inplace=True)

    #print(combined_data.describe())
    print(combined_data.head())
    print(combined_data.columns)

    combined_data.dropna(inplace=True)
    
    print ("Number of Unique Contacts: ",combined_data.contactId.nunique())
    print ("Number of Unique DateTimes",combined_data.created.nunique())

    print ("Step 1")

    # Group By Contact
    combined_data_t = combined_data ##
    
    combined_data_t.created = pd.to_datetime(combined_data.created, errors='coerce', utc = True)
    combined_data_t.dropna(subset = ['created'],inplace=True)
    
    # Calculating last day, last week and total count
    unique_ids = list(combined_data.contactId.unique())
    combined_data_t2 = combined_data_t.groupby('contactId').resample('D',on='created').count()['created']

    print (unique_ids[:1])

    lastday_count = []
    lastweek_count = []
    total_count = []

    unique_ids = np.array(unique_ids)
    unique_ids = unique_ids[~np.isnan(unique_ids)]
    #unique_ids = list(np.array(unique_ids)[~np.isnan(unique_ids)])

    for id in unique_ids:
        #print (id)
        lastday_count.append(combined_data_t2.loc[id].iloc[-1])
        lastweek_count.append(sum(combined_data_t2.loc[id].iloc[-7:-1]))
        total_count.append(sum(combined_data_t2.loc[id].iloc[0:-1]))
    
    

    print ("Step 2")

    train_data = pd.DataFrame()
    train_data['user_id'] = list(unique_ids)
    train_data['lastday_count'] = lastday_count
    train_data['lastweek_count'] = lastweek_count
    train_data['total_count'] = total_count


    train_data.to_csv('train_data_jan.csv',index=False)


    combined_data_t.dropna(inplace=True)
    count  = combined_data_t.groupby(['contactId','type']).agg('count').unstack(fill_value=0).stack()

    print ("Step 3")

    # Calculating count statistics for different types of activities
    EmailBounceback_count = []
    EmailClickthrough_count = []
    EmailOpen_count = []
    EmailSend_count = []
    FormSubmit_count = []
    PageView_count = []
    WebsiteVisit_count = []
    Subscribe_count = []
    Unsubscribe_count = []


    for id in unique_ids:

        EmailBounceback_count.append(int(count.loc[id].iloc[0]))
        EmailClickthrough_count.append(int(count.loc[id].iloc[1]))
        EmailOpen_count.append(int(count.loc[id].iloc[2]))
        EmailSend_count.append(int(count.loc[id].iloc[3]))
        FormSubmit_count.append(int(count.loc[id].iloc[4]))
        PageView_count.append(int(count.loc[id].iloc[5]))
        Subscribe_count.append(int(count.loc[id].iloc[6]))
        Unsubscribe_count.append(int(count.loc[id].iloc[6]))
        WebsiteVisit_count.append(int(count.loc[id].iloc[7]))

    print ("Step 4")

    train_data['EmailBounceback_count'] = EmailBounceback_count
    train_data['EmailClickthrough_count'] = EmailClickthrough_count
    train_data['EmailOpen_count'] = EmailOpen_count
    train_data['EmailSend_count'] = EmailSend_count
    train_data['FormSubmit_count'] = FormSubmit_count
    train_data['PageView_count'] = PageView_count
    train_data['Subscribe_count'] = Subscribe_count
    train_data['Unsubscribe_count'] = Unsubscribe_count
    train_data['WebsiteVisit_count'] = WebsiteVisit_count
    
    return(combined_data, train_data)
    
    
    
combined_data,train_data = processing (filter_data2)

train_data.to_csv('train_data_all_cams.csv',index=False)


train_data = pd.read_csv("train_data_all_cams.csv")

# Adding Dates
all_dates = []

for id in train_data['user_id']:
    a = combined_data.loc[combined_data["contactId"] == id].created.iloc[0]
    #print (a)
    all_dates.append(a)
    
train_data['date'] = all_dates
print ("-------------")

train_data.date = pd.to_datetime(train_data.date)
train_data.date = [i.date() for i in train_data.date]

train_data.to_csv('train_data_all_cams_full.csv',index=False)


train_data = pd.read_csv("train_data_all_cams_full.csv")
train_data.set_index('user_id', inplace=True)


test_size = len(train_data[train_data['date'] > '2019-05-01'])/len(train_data)
print ("Test Size :",test_size)

combined_data_mod1 = pd.DataFrame(train_data)
combined_data_mod1.created = pd.to_datetime(train_data.date)
#combined_data_mod1.created = [i.date() for i in train_data.date]

print ("Latest Date : ",max(combined_data_mod1.created))
print ("Oldest Date : ",min(combined_data_mod1.created))


combined_data_mod11 = pd.DataFrame(train_data)
combined_data_mod11.created = pd.to_datetime(train_data.date)
#combined_data_mod1.created = [i.date() for i in train_data.date]

print ("Latest Date : ",max(combined_data_mod11.created))
print ("Oldest Date : ",min(combined_data_mod11.created))



train_data2 = pd.read_csv("train_data_all_cams.csv")

# Adding Dates
all_dates2 = []

for id in train_data2['user_id']:
    a = combined_data.loc[combined_data["contactId"] == id].created.iloc[-1]
    #print (a)
    all_dates2.append(a)
    
train_data2['date'] = all_dates2
print ("-------------")

train_data2.date = pd.to_datetime(train_data2.date)
train_data2.date = [i.date() for i in train_data2.date]


combined_data_mod12 = pd.DataFrame(train_data2)
combined_data_mod12.created = pd.to_datetime(train_data2.date)



from sklearn.metrics import confusion_matrix

# Setting Form Submit values to either 0 or 1
train_data.FormSubmit_count = [1 if i > 0  else 0 for i in train_data.FormSubmit_count]


# Split the data into 20% test and 80% training
y = train_data['FormSubmit_count']
X = train_data.drop(['FormSubmit_count','date'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,shuffle=False, random_state=0)


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

print (train_data.FormSubmit_count.value_counts())

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=200,  max_depth=100,random_state=0, n_jobs=-1,class_weight={0:1,1:4})

# Train the classifier
clf.fit(X_train, y_train)


# Print the name and gini importance of each feature
for feature in zip(list(X_train), clf.feature_importances_):
    print(feature)
    
    
# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)
y_probab = clf.predict_proba(X_test)

# View The Accuracy Of Our Full Feature Model
print ("The Accuracy of Random Forest is :",accuracy_score(y_test, y_pred))


tn, fp, fn, tp = confusion_matrix( y_pred,y_test).ravel()
print (tn, fp, fn, tp)


hd = X_train.shape[0]
tl = X_test.shape[0]
print (X_train.shape[0])
print (X_test.shape[0])


output_train = pd.DataFrame()
output_train['id'] = list(y_train.index)
output_train['train'] = list(y_train)

output_test = pd.DataFrame()
output_test['id'] = list(y_test.index)
output_test['real'] = list(y_test)
output_test['predict'] = y_pred
output_test['probability'] = [max(i) for i in y_probab]

output_train['date'] = list(train_data.head(hd)['date'].values)

output_test['date'] = list(train_data.tail(tl)['date'].values) 


output_train.date = pd.to_datetime(output_train.date)
output_train.date = [i.date() for i in output_train.date]

output_test.date = pd.to_datetime(output_test.date)
output_test.date = [i.date() for i in output_test.date]

output_test = output_test[output_test['predict'] == 1]
output_test[output_test['real'] != output_test['predict']].to_csv('Predictions_Probab.csv',index=False)



tn, fp, fn, tp = confusion_matrix( y_pred,y_test).ravel()
print (tn, fp, fn, tp)




average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_pred)



unfilter_data= pd.concat([filter_data11,filter_data12],ignore_index=True, sort=True)

total_records_uf = len(unfilter_data)
total_ids_uf = unfilter_data.contactId.nunique()
min_day_uf = pd.to_datetime(unfilter_data.created.min())
max_day_uf = pd.to_datetime(unfilter_data.created.max())

print("Total Number of Records:",total_records_uf)
print("Total Number of Unique ids:",total_ids_uf)
print("Total Number of days of data:", (max_day_uf - min_day_uf).days)
print("Records span from ", min_day_uf.strftime("%B %d, %Y"), " to ", max_day_uf.strftime("%B %d, %Y"))


filter_data1= pd.concat([filter_data11,filter_data12],ignore_index=True, sort=True)
print ("Total Records:",len(filter_data1))


total_records_f = len(filter_data1)
total_ids_f = filter_data1.contactId.nunique()
min_day_f = pd.to_datetime(filter_data1.created.min())
max_day_f = pd.to_datetime(filter_data1.created.max())

print("Total Number of Records:",total_records_f)
print("Total Number of Unique ids:",total_ids_f)
print("Total Number of days of data:", (max_day_f - min_day_f).days)
print("Records span from ", min_day_f.strftime("%B %d, %Y"), " to ", max_day_f.strftime("%B %d, %Y"))



class MyResources(Resources):
    @property
    def css_raw(self):
        return super().css_raw + [
            """.bk-root {
                    background-color: whitesmoke;
                    border-color: #000000;
                    }
            """
        ]


curdoc().clear()
#reset_output()
#output_notebook()
#output_file("div.html")


text1 = """<h2>Un-Filtered Data Dimensions</h2><br>Total Number of Records:&ensp;&ensp;""" + str(total_records_uf)+"""<br>Total Number of Unique ids:&ensp;&ensp;""" + str(total_ids_uf) + "<br>Total Number of days of data:&ensp;"+ str((max_day_uf - min_day_uf).days) + "<br>Records span from "+ str(min_day_uf.strftime("%B %d, %Y"))+ " to "+ str(max_day_uf.strftime("%B %d, %Y"))
div1 = Div(text=text1,
width=340, height=140,style=dict([("padding", "8px 8px 8px 8px"),("background-color","#5186db"),("color","#efefef")]))

text3 = """<h2>Filtered Data Dimensions</h2><br>Total Number of Records: &ensp;&ensp; """ + str(total_records_f)+ """<br>Total Number of Unique ids:&ensp;&ensp;""" + str(total_ids_f) + "<br>Total Number of days of data: &ensp;"+ str((max_day_f - min_day_f).days) + "<br>Records span from "+ str(min_day_f.strftime("%B %d, %Y"))+ " to "+ str(max_day_f.strftime("%B %d, %Y"))
div3 = Div(text=text3,
width=340, height=140,style=dict([("padding", "8px 8px 8px 8px"),("background-color","#5186db"),("color","#efefef")]))

divm2 = Div(text="""<br>""",
width=100, height=150)
div2 = Div(text="""<br>""",
width=200, height=150)


#########################################################################################

text_4 = "<h2>Number of Unique Values</h2><br> "
div_4 = Div(text=text_4,
width=1450, height=25)



"""text_x1 = "<h4>Email Open</h4>"
div_x1 = Div(text=text_x1,
width=250, height=10)


text_x2 = "<h4>Email Send</h4>"
div_x2 = Div(text=text_x2,
width=250, height=10)



d_x = data = dict(
        dates=email_open.nunique().index.tolist(),
        downloads=email_open.nunique().values.tolist(),
    )
s_1 = ColumnDataSource(d_x)

c_1 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex = DataTable(source=s_1, columns=c_1,index_position=None, width=210, height=200)


d_x2 = data = dict(
        dates=email_send.nunique().index.tolist(),
        downloads=email_send.nunique().values.tolist(),
    )
s_2 = ColumnDataSource(d_x2)

c_2 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex2 = DataTable(source=s_2, columns=c_2,index_position=None, width=210, height=200)


###############################################
text_x3 = "<h4>Email Click Through</h4>"
div_x3 = Div(text=text_x3,
width=250, height=10)



d_x3 = data = dict(
        dates=email_click.nunique().index.tolist(),
        downloads=email_click.nunique().values.tolist(),
    )
s_3 = ColumnDataSource(d_x3)

c_3 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex3 = DataTable(source=s_3, columns=c_3,index_position=None, width=210, height=200)


###############################################
text_x4 = "<h4>Web Visit</h4>"
div_x4 = Div(text=text_x4,
width=250, height=10)



d_x4 = data = dict(
        dates=web_visit.nunique().index.tolist(),
        downloads=web_visit.nunique().values.tolist(),
    )
s_4 = ColumnDataSource(d_x4)

c_4 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex4 = DataTable(source=s_4, columns=c_4,index_position=None, width=210, height=200)


###############################################
text_x5 = "<h4>Email Bounceback</h4>"
div_x5 = Div(text=text_x5,
width=250, height=10)



d_x5 = data = dict(
        dates=bounceback.nunique().index.tolist(),
        downloads=bounceback.nunique().values.tolist(),
    )
s_5 = ColumnDataSource(d_x5)

c_5 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex5 = DataTable(source=s_5, columns=c_5,index_position=None, width=210, height=200)



###############################################
text_x6 = "<h4>Form Submit</h4>"
div_x6 = Div(text=text_x6,
width=250, height=10)



d_x6 = data = dict(
        dates=form_submit.nunique().index.tolist(),
        downloads=form_submit.nunique().values.tolist(),
    )
s_6 = ColumnDataSource(d_x6)

c_6 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex6 = DataTable(source=s_6, columns=c_6,index_position=None, width=210, height=200)



###############################################
text_x7 = "<h4>Page View</h4>"
div_x7 = Div(text=text_x7,
width=250, height=10)



d_x7 = data = dict(
        dates=pageview.nunique().index.tolist(),
        downloads=pageview.nunique().values.tolist(),
    )
s_7 = ColumnDataSource(d_x7)

c_7 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex7 = DataTable(source=s_7, columns=c_7,index_position=None, width=210, height=200)


###############################################
text_x8 = "<h4>Subscribe</h4>"
div_x8 = Div(text=text_x8,
width=250, height=10)



d_x8 = data = dict(
        dates=subscribe.nunique().index.tolist(),
        downloads=subscribe.nunique().values.tolist(),
    )
s_8 = ColumnDataSource(d_x8)

c_8 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex8 = DataTable(source=s_8, columns=c_8,index_position=None, width=210, height=200)



###############################################
text_x9 = "<h4>Unsubscribe</h4>"
div_x9 = Div(text=text_x9,
width=250, height=10)



d_x9 = data = dict(
        dates=unsubscribe.nunique().index.tolist(),
        downloads=unsubscribe.nunique().values.tolist(),
    )
s_9 = ColumnDataSource(d_x9)

c_9 = [
        TableColumn(field="dates", title="Value"),
        TableColumn(field="downloads", title="Count"),
    ]
data_tablex9 = DataTable(source=s_9, columns=c_9,index_position=None, width=210, height=200)"""
###################################################################################################################################

def func_a(name,list_a,list_b):
    
    texta = """<html>
    <head>
    <style>
    table {
      width:100%%;
    }
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
    }
    th, td {
      padding: 5px;
      text-align: left;
    }
    table#t01 tr:nth-child(even) {
      background-color: white;
    }
    table#t01 tr:nth-child(odd) {
     background-color: white;
    }
    table#t01 th {
      background-color: black;
      color: white;
    }
    </style>
    </head>
    <body>

    <h3>%s</h3>

    <table id="t01">
      <tr>
        <th>Value</th>
        <th>Count</th> 
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>

    </table>

    </body>
    </html>""" % (name,list_a[0],list_b[0],list_a[1],list_b[1],list_a[2],list_b[2],list_a[3],list_b[3],list_a[4],list_b[4],list_a[5],list_b[5])
    
    return (texta)


def func_b(name,list_a,list_b):
    
    textb = """<html>
    <head>
    <style>
    table {
      width:100%%;
    }
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
    }
    th, td {
      padding: 5px;
      text-align: left;
    }
    table#t01 tr:nth-child(even) {
      background-color: white;
    }
    table#t01 tr:nth-child(odd) {
     background-color: white;
    }
    table#t01 th {
      background-color: black;
      color: white;
    }
    </style>
    </head>
    <body>

    <h3>%s</h3>

    <table id="t01">
      <tr>
        <th>Value</th>
        <th>Count</th> 
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>
      <tr>
        <td>%s</td>
        <td>%d</td>
      </tr>

    </table>

    </body>
    </html>""" % (name,list_a[0],list_b[0],list_a[1],list_b[1],list_a[2],list_b[2],list_a[3],list_b[3])
    
    return (textb)

## ,style=dict([("padding", "10px 20px 10px 20px"),("background-color","white")])

diva = Div(text=func_a('Email Open', email_open.nunique().index.tolist(), email_open.nunique().values.tolist()),
width=200, height=250)

divb = Div(text=func_a('Email Send', email_send.nunique().index.tolist(), email_send.nunique().values.tolist()),
width=200, height=250)

divc = Div(text=func_a('Email Click Through', email_clickthrough.nunique().index.tolist(), email_clickthrough.nunique().values.tolist()),
width=200, height=250)

divd = Div(text=func_a('Web Visit', web_visit.nunique().index.tolist(), web_visit.nunique().values.tolist()),
width=200, height=250)

dive = Div(text=func_a('Email Bounceback', bounceback.nunique().index.tolist(), bounceback.nunique().values.tolist()),
width=200, height=250)

divf = Div(text=func_b('Form Submit', form_submit.nunique().index.tolist(), form_submit.nunique().values.tolist()),
width=200, height=200)

divg = Div(text=func_b('Page View', pageview.nunique().index.tolist(), pageview.nunique().values.tolist()),
width=200, height=200)

divh = Div(text=func_b('Subscribe', subscribe.nunique().index.tolist(), subscribe.nunique().values.tolist()),
width=200, height=200)

divi = Div(text=func_b('UnSubscribe', unsubscribe.nunique().index.tolist(), unsubscribe.nunique().values.tolist()),
width=200, height=200)

#show(gridplot([[widgetbox(diva),widgetbox(divb),widgetbox(divc),widgetbox(divd),widgetbox(dive)],[widgetbox(divf),widgetbox(divg),widgetbox(divh),widgetbox(divi)]]))
#show(column(row(widgetbox(diva),widgetbox(divb),widgetbox(divc),widgetbox(divd),widgetbox(dive)), row(widgetbox(divf),widgetbox(divg),widgetbox(divh),widgetbox(divi))))




div_a = row(divm2,div1,div2, div3)
div_c = row(div_4)
#row_x = row(div_x1,div_x2,div_x3,div_x4,div_x5)
#row_x1 = row(widgetbox(data_tablex,width=250),widgetbox(data_tablex2,width=250),widgetbox(data_tablex3,width=250),widgetbox(data_tablex4,width=250),widgetbox(data_tablex5,width=250)) 
#row_x2 = row(div_x6,div_x7,div_x8,div_x9)
#row_x3 = row(widgetbox(data_tablex6,width=250),widgetbox(data_tablex7,width=250),widgetbox(data_tablex8,width=250),widgetbox(data_tablex9,width=250)) 

row_x1 = row(widgetbox(diva),widgetbox(divb),widgetbox(divc),widgetbox(divd),widgetbox(dive))
row_x2 =  row(widgetbox(divf),widgetbox(divg),widgetbox(divh),widgetbox(divi))

div_all  = column(div_a,div_c,row_x1,row_x2)

tab1 = Panel(child=div_all, title="About Data")


tabs1 = Tabs(tabs=[ tab1 ])

##tb = Panel(child=tabs, title="About Data")
##tbs = Tabs(tabs = [tb])

#show(tabs1)

#show(widgetbox(div))
#show(widgetbox(div2))



#################################################



combined_data_mod1 = pd.DataFrame(combined_data)
combined_data_mod1.created = pd.to_datetime(combined_data.created)
combined_data_mod1.created = [i.date() for i in combined_data_mod1.created]


combined_data_mod2 = pd.DataFrame(combined_data)
combined_data_mod2.created = pd.to_datetime(combined_data.created)
combined_data_mod2.created = [i.weekday_name for i in combined_data_mod2.created]




curdoc().clear()
#output_file("Output_Jan.html")

#reset_output()
#output_notebook()

text1 = """<h2>Un-Filtered Data Dimensions</h2><br>Total Number of Records:&ensp;&ensp;""" + str(total_records_uf)+"""<br>Total Number of Unique ids:&ensp;&ensp;""" + str(total_ids_uf) + "<br>Total Number of days of data:&ensp;"+ str((max_day_uf - min_day_uf).days) + "<br>Records span from "+ str(min_day_uf.strftime("%B %d, %Y"))+ " to "+ str(max_day_uf.strftime("%B %d, %Y"))
div1 = Div(text=text1,
width=340, height=140,style=dict([("padding", "8px 8px 8px 8px"),("background-color","#5186db"),("color","#efefef")]))

text3 = """<h2>Filtered Data Dimensions</h2><br>Total Number of Records: &ensp;&ensp; """ + str(total_records_f)+ """<br>Total Number of Unique ids:&ensp;&ensp;""" + str(total_ids_f) + "<br>Total Number of days of data: &ensp;"+ str((max_day_f - min_day_f).days) + "<br>Records span from "+ str(min_day_f.strftime("%B %d, %Y"))+ " to "+ str(max_day_f.strftime("%B %d, %Y"))
div3 = Div(text=text3,
width=340, height=140,style=dict([("padding", "8px 8px 8px 8px"),("background-color","#5186db"),("color","#efefef")]))

divm2 = Div(text="""<br>""",
width=100, height=150)
div2 = Div(text="""<br>""",
width=200, height=150)



#Text Div

text_div1 = Div(text="""<br><br><br><br><br><br><br>This graph shows a histogram for different types of Campaign Activities.""",
 style={'font-size': '115%'},width=200, height=450)

text_div2 = Div(text="""<br><br><br><br><br>This graph shows the activity count by week. The x-axis signify the week from now i.e the 0 signify the last week of activity and 1 signify the second last week of activity. This value is calculated by averaging the number of acitivities done by different Contact Ids over their activity span.""",
style={'font-size': '115%'},width=200, height=450)

text_div3 = Div(text="""<br><br><br><br><br><br><br><br>This graph shows a boxplot.<i> A box plot is a graphical rendition of statistical data based on the minimum, first quartile, median, third quartile, and maximum.</i> The boxes show the distribution from 25 precentile to 75 percentile. The bars on top and bottom of the boxes show outlier range. Dots outside these bars show us the spread of the outliers.""",
style={'font-size': '115%'},width=200, height=450)

text_div4 = Div(text="""<br><br><br><br><br><br><br><br>This graph shows a multiline graph for different activity types plotted across different Dates.""",
style={'font-size': '115%'},width=200, height=450)

text_div42 = Div(text="""<br><br><br><br><br><br><br><br>This graph shows the number of unique Contact IDs campaigned to across different Dates.""",
style={'font-size': '115%'},width=200, height=450)

text_div43 = Div(text="""<br><br><br><br><br><br><br><br>This graph shows the number of Form_Submits made across different Dates.""",
style={'font-size': '115%'},width=200, height=450)

text_div44 = Div(text="""<br><br><br><br><br><br><br><br>This graph shows the number of No-Form_Submits made across different Dates.""",
style={'font-size': '115%'},width=200, height=450)

text_div5 = Div(text="""<br><br><br><br><br><br><br><br>This graph shows a histogram for different Email Send Types.""",
style={'font-size': '115%','vertical-align': 'middle'},width=200, height=450)

text_div6 = Div(text="""<br><br><br><br><br><br><br><br>This graph shows a stacked bar graphs for different acitivity types across weekdays. <i>Hover over the bar graph to get exact count.</i>""",
style={'font-size': '115%'},width=200, height=450)



#P1
fruits1 = list(dict(email_open.emailSendType.value_counts().sort_index()).keys())
counts1 = list(dict(email_open.emailSendType.value_counts().sort_index()).values())

source1 = ColumnDataSource(data=dict(fruits1=fruits1, counts1=counts1))

p1 = figure(x_range=fruits1, plot_height=300,plot_width=300, title="Email Send Type",tools="pan,wheel_zoom,box_zoom,save,reset")
p1.vbar(x='fruits1',  width=0.3, top='counts1', source=source1,
       line_color='white', fill_color=factor_cmap('fruits1', palette=Spectral6, factors=fruits1))
#p.hbar(y=fruits,  height=0.2, left=0,right=counts, color="#CAB2D6")
p1.xgrid.grid_line_color = None
p1.y_range.start = 0
p1.y_range.end = 75000
p1.xaxis.major_label_orientation = pi/4
##p1.border_fill_color = "white"
##p1.min_border = 15
#p1.legend.label_text_font_size = '7pt'
#new_legend = p1.legend[0]
#p1.legend[0].plot = None
#p1.add_layout(new_legend, 'right')
#p1.xaxis.major_label_orientation = "vertical"


#P2
data = dict(combined_data['type'].value_counts().sort_index())
source2 = ColumnDataSource(dict(y=list(data.keys()), right=list(data.values()),))

p2 = figure( title="Activities Count",plot_width=520, plot_height=350,  y_range=list(data.keys()), x_range=(0, max(data.values()) * 1.1),tools="pan,wheel_zoom,box_zoom,save,reset")
p2.hbar(y='y', height=0.7, left=0, right='right',color='#86bf91',source=source2)
p2.yaxis.axis_label = 'Activities'
p2.xaxis.axis_label = 'Count'
p2.xaxis.major_label_orientation = pi/4
##p2.border_fill_color = "white"
##p2.min_border = 35

#P4
def color_gen():
    yield from itertools.cycle(Category10[10])
color = color_gen()

p4 = figure(plot_width=830, plot_height=350,title='Activity Count by Date',x_axis_label='Date', y_axis_label='Number of Activities',x_axis_type="datetime",tools="pan,wheel_zoom,box_zoom,save,reset")

for t,c in zip(combined_data.type.unique(),color):
    
    row1  = list(combined_data_mod1[combined_data_mod1.type == t].created.value_counts().sort_index().index)
    column1 = list(combined_data_mod1[combined_data_mod1.type == t].created.value_counts().sort_index().values)
    p4.line(x = row1, y = column1, color=c, legend = t)

p4.legend.location = "top_left"
p4.legend.label_text_font_size = '7pt'
p4.legend.click_policy="hide"
##p4.border_fill_color = "white"
##p4.min_border = 35

#P5
colors = ["green", "yellow", "red",'grey', '#30678D', '#35B778','beige', '#FDE724','#a1dab4']

df1 = pd.crosstab(combined_data_mod2.created , combined_data_mod2.type)
source5 = ColumnDataSource(df1)

day = source5.data['created'].tolist()
p5 = figure(x_range= day,plot_width=400 ,plot_height=600, toolbar_location=None, tools="hover", tooltips="$name: @$name")

p5.vbar_stack(stackers=combined_data.type.unique(), 
             x='created', source=source5, 
             legend = [value(x) for x in combined_data.type.unique()],
             width=0.5, color=colors)

p5.title.text ='Activities Split by Weekday'
p5.xaxis.axis_label = 'Day'
p5.xgrid.grid_line_color = None	#remove the x grid lines
p5.yaxis.axis_label = 'Activity Type'
p5.xaxis.major_label_orientation = pi/4
p5.legend.location = "top_right"
p5.legend.orientation = "vertical"
p5.legend.label_text_font_size = '7pt'
new_legend = p5.legend[0]
p5.legend[0].plot = None
p5.add_layout(new_legend, 'right')
##p5.border_fill_color = "white"
##p5.min_border = 35

#P6
fruits6 = list(train_data.lastweek_count.value_counts()[:10].index)
counts6 = list(train_data.lastweek_count.value_counts()[:10].values)
fruits6.sort()

source6 = ColumnDataSource(data=dict(fruits6=fruits6, counts6=counts6, color=Category10[10]))
p6 = figure( plot_height=300,plot_width=300, title="Activity Count by Week",tools="pan,wheel_zoom,box_zoom,save,reset")
p6.vbar(x='fruits6', top='counts6', width=0.4, color='color', source=source6)
p6.xgrid.grid_line_color = None
p6.xaxis.axis_label = 'Count'
p6.xgrid.grid_line_color = None	#remove the x grid lines
p6.ygrid.grid_line_color = None
p6.yaxis.axis_label = 'Activity Span'
##p6.border_fill_color = "white"
##p6.min_border = 15

# P7
p7 = figure(plot_width=600 ,plot_height=600,title="Activity Distribution",x_range=(0,1), y_range=(0,1),tools="pan,wheel_zoom,box_zoom,save,reset")
p7.image_url(url=['fabretto/static/img1.png'],x=0.12, y=1, w=0.7, h=1)
p7.xgrid.grid_line_color = None	#remove the x grid lines
p7.ygrid.grid_line_color = None
##p7.border_fill_color = "white"
##p7.min_border = 15
##p7.min_border_left = 15

#p42
p42 = figure(plot_width=750, plot_height=450,title='Contact ID Count vs Date',x_axis_label='Date', y_axis_label='Contact ID Count',x_axis_type="datetime")
p42.line(x = list(output_train.date.value_counts().sort_index().index), y = list(output_train.date.value_counts().sort_index().values), color="blue", legend='train')
p42.line(x = list(output_test.date.value_counts().sort_index().index), y = list(output_test.date.value_counts().sort_index().values), color="orange", legend='pred')
#p42.legend.location = "top_left"
p42.legend.location = "top_right"
p42.legend.label_text_font_size = '7pt'



#p43
p43 = figure(plot_width=750, plot_height=450,title='Form Submit Count vs Date',x_axis_label='Date', y_axis_label='Form Submit Count',x_axis_type="datetime")
p43.line(x = list(output_train[output_train['train'] == 1].date.value_counts().sort_index().index), y = list(output_train[output_train['train'] == 1].date.value_counts().sort_index().values), color="blue", legend='train')
p43.line(x = list(output_test[output_test['predict'] == 1].date.value_counts().sort_index().index), y = list(output_test[output_test['predict'] == 1].date.value_counts().sort_index().values), color="orange", legend='prediction')
#p43.legend.location = "top_left"
p43.legend.location = "top_right"
p43.legend.label_text_font_size = '7pt'



#p44
p44 = figure(plot_width=750, plot_height=450,title='No Form Submit Count vs Date',x_axis_label='Date', y_axis_label='No Form Submit Count',x_axis_type="datetime")
p44.line(x = list(output_train[output_train['train'] == 0].date.value_counts().sort_index().index), y = list(output_train[output_train['train'] == 0].date.value_counts().sort_index().values), color="blue", legend='train')
p44.line(x = list(output_test[output_test['predict'] == 0].date.value_counts().sort_index().index), y = list(output_test[output_test['predict'] == 0].date.value_counts().sort_index().values), color="orange", legend='pred')
#p44.legend.location = "top_left"
p44.legend.location = "top_right"
p44.legend.label_text_font_size = '7pt'


## New ##
#p72
p72 = figure(plot_width=400, plot_height=350,title='Contact ID Count vs Start Date',x_axis_label='Start Date', y_axis_label='Contact ID Count',x_axis_type="datetime",tools="pan,wheel_zoom,box_zoom,save,reset")
p72.line(x = list(combined_data_mod11.created.value_counts().sort_index().index), y = list(combined_data_mod11.created.value_counts().sort_index().values), color="blue")
#p23.legend.location = "top_left"
#p23.legend.location = "top_right"
#p23.legend.label_text_font_size = '7pt'
p72.border_fill_color = "white"
p72.min_border = 15


#p73
p73 = figure(plot_width=400, plot_height=350,title='Contact ID Count vs End Date',x_axis_label='End Date', y_axis_label='Contact ID Count',x_axis_type="datetime",tools="pan,wheel_zoom,box_zoom,save,reset")
p73.line(x = list(combined_data_mod12.created.value_counts().sort_index().index), y = list(combined_data_mod12.created.value_counts().sort_index().values), color="blue")
#p23.legend.location = "top_left"
#p23.legend.location = "top_right"
#p23.legend.label_text_font_size = '7pt'
p73.border_fill_color = "white"
p73.min_border = 15


x = ((combined_data_mod12.created - combined_data_mod11.created).value_counts()[:50]).sort_index()

#p74
list_x74 = list((x.sort_index().index)/(3600000*24))
list_y74 = list(x.sort_index().values)
source74 = ColumnDataSource(data=dict(x=list_x74, y=list_y74))

p74 = figure(plot_width=500, plot_height=350,title='Duration of Campaign - Number of Days vs Count',x_axis_label='Number of Days', y_axis_label='Count',tools="hover,pan,wheel_zoom,box_zoom,save,reset", tooltips=[("Days", "@x"),("Count", "@y")])
p74.line('x','y' , color="blue",source = source74)
p74.circle('x','y', color="green",source = source74)
#p23.legend.location = "top_left"
#p23.legend.location = "top_right"
#p23.legend.label_text_font_size = '7pt'
p74.border_fill_color = "white"
p74.min_border = 1



#plt.figure(figsize=(16,8))
#combined_data_mod1.created.value_counts().plot.line()
#plt.xlabel("Date")
#plt.ylabel("Counts")
#plt.title("Fig 16")
#plt.show()



colors = ["green", "yellow", "red",'grey', '#30678D', '#35B778','beige', '#FDE724','#a1dab4']

df75 = pd.crosstab(combined_data_mod3.created , combined_data_mod3.type)
source75 = ColumnDataSource(df75)

day = source75.data['created'].tolist()
p75 = figure(plot_width=1400 ,plot_height=400, toolbar_location=None, tools="hover", tooltips="$name: @$name")

p75.vbar_stack(stackers=combined_data.type.unique(), 
             x='created', source=source75, 
             legend = [value(x) for x in combined_data.type.unique()],
             width=0.5, color=colors)

p75.title.text ='Activities Split by Day of Month'
p75.xaxis.axis_label = 'Day of Month'
p75.xgrid.grid_line_color = None	#remove the x grid lines
p75.yaxis.axis_label = 'Activity Type'
p75.xaxis.major_label_orientation = pi/4
p75.legend.location = "top_right"
p75.legend.orientation = "vertical"
p75.legend.label_text_font_size = '7pt'
new_legend = p75.legend[0]
p75.legend[0].plot = None
p75.add_layout(new_legend, 'right')
##p5.border_fill_color = "white"
##p5.min_border = 35



## New ##





div251 = Div(text="",width=50, height=350)
div252 = Div(text="",width=50, height=20)
div253 = Div(text="",width=50, height=20)
div254 = Div(text="",width=50, height=20)
div25 = Div(text="",width=50, height=20)
div26 = Div(text="",width=50, height=600)
div27 = Div(text="",width=50, height=20)
div28 = Div(text="",width=50, height=20)
div29 = Div(text="",width=50, height=600)
div291 = Div(text="",width=50, height=350)
div292 = Div(text="",width=50, height=350)
div293 = Div(text="",width=280, height=20)
div294 = Div(text="",width=280, height=20)
div295 = Div(text="",width=280, height=20)
div296 = Div(text="",width=280, height=20)
div297 = Div(text="",width=280, height=20)
div298 = Div(text="",width=280, height=20)
div299 = Div(text="",width=280, height=20)

#r1 = row(p1,div5,p2)
#r2 = row(p4,div5,p6)
#r3 = row(p5,div5,p7)
col1 = column(p6,p1)

#tab2 = Panel(child=col1, title="Analysis")
#tabs2 = Tabs(tabs=[tab2])

# make a grid
grid = gridplot([[divm2,div1,div2,div3],[p2,div251,p4] ,[p7, div26,col1, div29,p5],[p72,div291,p73,div292,p74],[p75]],merge_tools = False,toolbar_options=dict(logo=None))
#[div297,text_div42,div252,p42], [div298,text_div43,div253,p43], [div299,text_div43,div253,p43]
# make a grid
#grid = gridplot([p2,p6,p7,p4,p1,p5], ncols=3, plot_width=450, plot_height=300)


tab2 = Panel(child=grid, title="Analysis")
tabs2 = Tabs(tabs=[tab2])


#show(tabs2)




curdoc().clear()

#reset_output()
#output_notebook()

text1 = """<h2>Un-Filtered Data Dimensions</h2><br>Total Number of Records:&ensp;&ensp;""" + str(total_records_uf)+"""<br>Total Number of Unique ids:&ensp;&ensp;""" + str(total_ids_uf) + "<br>Total Number of days of data:&ensp;"+ str((max_day_uf - min_day_uf).days) + "<br>Records span from "+ str(min_day_uf.strftime("%B %d, %Y"))+ " to "+ str(max_day_uf.strftime("%B %d, %Y"))
div1 = Div(text=text1,
width=340, height=140,style=dict([("padding", "8px 8px 8px 8px"),("background-color","#5186db"),("color","#efefef")]))

text3 = """<h2>Filtered Data Dimensions</h2><br>Total Number of Records: &ensp;&ensp; """ + str(total_records_f)+ """<br>Total Number of Unique ids:&ensp;&ensp;""" + str(total_ids_f) + "<br>Total Number of days of data: &ensp;"+ str((max_day_f - min_day_f).days) + "<br>Records span from "+ str(min_day_f.strftime("%B %d, %Y"))+ " to "+ str(max_day_f.strftime("%B %d, %Y"))
div3 = Div(text=text3,
width=340, height=140,style=dict([("padding", "8px 8px 8px 8px"),("background-color","#5186db"),("color","#efefef")]))

divm2 = Div(text="""<br>""",
width=100, height=150)
div2 = Div(text="""<br>""",
width=200, height=150)


#P8
"""p8 = figure(plot_width=500,plot_height=400,title="Audience Correlation",x_range=(0,1), y_range=(0,1))
p8.image_url(url=['img2.png'],x=0.02, y=1, w=0.9, h=0.9)
p8.xgrid.grid_line_color = None	#remove the x grid lines
p8.ygrid.grid_line_color = None
p8.border_fill_color = "whitesmoke"
p8.min_border = 35"""

corr_table =train_data.iloc[:,1:].corr()

years = list(corr_table.index)
months = list(corr_table.columns)

df = pd.DataFrame(corr_table.stack(), columns=['rate']).reset_index()


# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
#mapper = LinearColorMapper(palette=colors, low=0, high=1)
mapper = LogColorMapper(palette="Magma256", low=df.rate.min(), high=df.rate.max())

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

p8 = figure(title="Audience Correlation",
           x_range=years, y_range=list(reversed(months)),
           x_axis_location="above", plot_width=575, plot_height=450,
           tools=TOOLS, toolbar_location='below',
           tooltips=[('correlation', '@rate')])

p8.grid.grid_line_color = None
p8.axis.axis_line_color = None
p8.axis.major_tick_line_color = None
p8.axis.major_label_text_font_size = "7pt"
p8.axis.major_label_standoff = 0
p8.xaxis.major_label_orientation = pi / 2

p8.rect(x="level_0", y="level_1", width=1, height=1,
       source=df,
       fill_color={'field': 'rate', 'transform': mapper},
       line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d%%"),
                     label_standoff=6, border_line_color=None, location=(0, 0))


p8.add_layout(color_bar, 'right')
p8.border_fill_color = "white"
p8.min_border = 35


#P9
pie_data = dict(zip(list(X_train), clf.feature_importances_))
pie_data = dict(sorted(pie_data.items(), key=itemgetter(1)))

dict(sorted(pie_data.items(), key=itemgetter(1)))



data9 = pd.Series(pie_data).reset_index(name='value').rename(columns={'index':'importance'})
data9['angle'] = data9['value']/data9['value'].sum() * 2*pi
data9['color'] = Category20c[len(pie_data)]

p9 = figure(plot_width=575,plot_height=450, title="Feature Importance", toolbar_location=None,
           tools="hover", tooltips="@importance: @value", x_range=(-0.5, 1.0))

p9.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend='importance', source=data9)

p9.axis.axis_label=None
p9.axis.visible=False
p9.grid.grid_line_color = None
p9.legend.label_text_font_size = '7pt'
p9.border_fill_color = "white"
p9.min_border = 35

#P10
results10 = ['Positives', 'Negatives']
years10 = ['True','False']
colors10 = ["#718dbf", "#e84d60"]

data10 = {'Results' : results10,
        'True'   : [tp,tn],
        'False'   : [fp,fn]}

p10 = figure(x_range=results10, plot_height=250,plot_width=400, title="Results",
           toolbar_location=None, tools="hover", tooltips="$name @Results: @$name")

p10.vbar_stack(years10, x='Results', width=0.9, color=colors10, source=data10,
             legend=[value(x) for x in years10])

p10.y_range.start = 0
p10.y_range.range_padding = 1
p10.x_range.range_padding = 0.1
p10.xgrid.grid_line_color = None
p10.axis.minor_tick_line_color = None
p10.outline_line_color = None
p10.legend.location = "top_left"
p10.legend.orientation = "horizontal"
p10.border_fill_color = "white"
p10.min_border = 20

#p20
p20 = figure(plot_width=400, plot_height=250,title = '2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision),x_range=(0, 1),y_range=(0, 1),tools="pan,wheel_zoom,box_zoom,save,reset")
p20.quad(top=[precision[1], precision[0]], bottom=[0, 0], left=[0, 0],
       right=[recall[1], recall[0]], color="#d8ceea")
p20.border_fill_color = "white"
p20.min_border = 20

#p22
p22 = figure(plot_width=750, plot_height=500,title='Contact ID Count vs Date',x_axis_label='Date', y_axis_label='Contact ID Count',x_axis_type="datetime",tools="pan,wheel_zoom,box_zoom,save,reset")
p22.line(x = list(output_train.date.value_counts().sort_index().index), y = list(output_train.date.value_counts().sort_index().values), color="blue", legend='train')
p22.line(x = list(output_test.date.value_counts().sort_index().index), y = list(output_test.date.value_counts().sort_index().values), color="orange", legend='pred')
#p22.legend.location = "top_left"
p22.legend.location = "top_right"
p22.legend.label_text_font_size = '7pt'
p22.border_fill_color = "white"
p22.min_border = 35

#p23
p23 = figure(plot_width=575, plot_height=400,title='Form Submit Count vs Date',x_axis_label='Date', y_axis_label='Form Submit Count',x_axis_type="datetime",tools="pan,wheel_zoom,box_zoom,save,reset")
p23.line(x = list(output_train[output_train['train'] == 1].date.value_counts().sort_index().index), y = list(output_train[output_train['train'] == 1].date.value_counts().sort_index().values), color="blue", legend='train')
p23.line(x = list(output_test[output_test['predict'] == 1].date.value_counts().sort_index().index), y = list(output_test[output_test['predict'] == 1].date.value_counts().sort_index().values), color="orange", legend='prediction')
#p23.legend.location = "top_left"
p23.legend.location = "top_right"
p23.legend.label_text_font_size = '7pt'
p23.border_fill_color = "white"
p23.min_border = 35

#p24
p24 = figure(plot_width=575, plot_height=400,title='No Form Submit Count vs Date',x_axis_label='Date', y_axis_label='No Form Submit Count',x_axis_type="datetime",tools="pan,wheel_zoom,box_zoom,save,reset")
p24.line(x = list(output_train[output_train['train'] == 0].date.value_counts().sort_index().index), y = list(output_train[output_train['train'] == 0].date.value_counts().sort_index().values), color="blue", legend='train')
p24.line(x = list(output_test[output_test['predict'] == 0].date.value_counts().sort_index().index), y = list(output_test[output_test['predict'] == 0].date.value_counts().sort_index().values), color="orange", legend='pred')
#p24.legend.location = "top_left"
p24.legend.location = "top_right"
p24.legend.label_text_font_size = '7pt'
p24.border_fill_color = "white"
p24.min_border = 35

div30 = Div(text="",
width=50, height=450)

div31 = Div(text="",
width=50, height=455)

div32 = Div(text="",
width=50, height=505)

div33 = Div(text="",
width=50, height=400)

r = row(p8,div31,p9)
rr = row(p10, div32, p20)
colt4 = column(p10, p20)

grid2 = gridplot([[divm2,div1,div2,div3],[p8,div31,p9],[colt4, div32,p22],[p23,div33,p24]],merge_tools = False,toolbar_options=dict(logo=None))

b_play = "<style>.custom{ background-color: #98FB98 }</style>"


tab3 = Panel(child=grid2, title="Model")
tabs3 = Tabs(tabs=[tab3], css_classes =['custom'])

#show(tabs3)


curdoc().clear()

#reset_output()
#output_notebook()

text40 = """<h3 align="center"></h3>"""
div40 = Div(text=text40,
width=250, height=20)

text41 = """<h3 align="center">All</h3>"""
div41 = Div(text=text41,
width=250, height=20)


text42 = """<h3 align="center">Form Submit No</h3>"""
div42 = Div(text=text42,
width=250, height=20)

text43 = """<h3 align="center">Form Submit Yes</h3>"""
div43 = Div(text=text43,
width=250, height=20)

dft4 = pd.DataFrame()
dft4['ContactID'] = list(pd.DataFrame(y_test).index)
dft4['FormSubmit'] = y_pred

dft4_0 = dft4[dft4['FormSubmit'] == 0]
dft4_1 = dft4[dft4['FormSubmit'] == 1]

# All
data_t4 = dict(
        ContactID=dft4.ContactID,
        FormSubmit=dft4.FormSubmit,
    )
source_t4 = ColumnDataSource(data_t4)

columns_t4 = [
        TableColumn(field="ContactID", title="Contact Id"),
        TableColumn(field="FormSubmit", title="Form Submit"),
    ]
data_table_t4 = DataTable(name="All",source=source_t4, columns=columns_t4, width=250, height=480)



#New Block - 14/2

df_alt = pd.read_csv(join(dirname(__file__),"Predictions_Probab_alt.csv"))

source = ColumnDataSource(data=dict())

def update():
    #current = df[(df['salary'] >= slider.value[0]) & (df['salary'] <= slider.value[1])].dropna()
    current = df_alt.loc[:slider.value[1]]
    source.data = {
        'id'             : current.id,
        'date'           : current.date,
        'predict'        : current.predict,
    }

slider = RangeSlider(title="Length", start=0, end=len(df_alt), value=(0, 100), step=2, format="0,0")
slider.on_change('value', lambda attr, old, new: update())

# All Download Button
callback1 = CustomJS(args=dict(source=source), code="""
var data = source.data;
var filetext = 'ContactID,FormSubmit\\n';

for (i=0; i < data['ContactID'].length; i++) {
    var currRow = [data['ContactID'][i].toString(), data['FormSubmit'][i].toString().concat('\\n')];
    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'all.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
""")
button1 = Button(label='Download', button_type='success', callback=callback1)


columns_t4 = [
        TableColumn(field="id", title="Id"),
        TableColumn(field="date", title="Date"),
        TableColumn(field="predict", title="Predict"),
    ]
data_table_t4 = DataTable(name="All",source=source, columns=columns_t4, width=250, height=480)


controls = column(slider, button1)
#curdoc().add_root(row(controls, data_table_t4))




## All Download Button
#callback1 = CustomJS(args=dict(source=source_t4), code="""
#var data = source.data;
#var filetext = 'ContactID,FormSubmit\\n';

#for (i=0; i < data['ContactID'].length; i++) {
#    var currRow = [data['ContactID'][i].toString(), data['FormSubmit'][i].toString().concat('\\n')];
#    var joined = currRow.join();
#    filetext = filetext.concat(joined);
#}

#var filename = 'all.csv';
#var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

#//addresses IE
#if (navigator.msSaveBlob) {
#    navigator.msSaveBlob(blob, filename);
#}

#else {
#    var link = document.createElement("a");
#    link = document.createElement('a')
#    link.href = URL.createObjectURL(blob);
#    link.download = filename
#    link.target = "_blank";
#    link.style.visibility = 'hidden';
#    link.dispatchEvent(new MouseEvent('click'))
#}
#""")
#button1 = Button(label='Download', button_type='success', callback=callback1)

###############################################################################################################################


# No
data_t40 = dict(
        ContactID=dft4_0.ContactID,
        FormSubmit=dft4_0.FormSubmit,
    )
source_t40 = ColumnDataSource(data_t40)

columns_t40 = [
        TableColumn(field="ContactID", title="Contact Id"),
        TableColumn(field="FormSubmit", title="Form Submit"),
    ]
data_table_t40 = DataTable(name="All",source=source_t40, columns=columns_t40, width=250, height=480)


# No Submit Download Button
callback2 = CustomJS(args=dict(source=source_t40), code="""
var data = source.data;
var filetext = 'ContactID,FormSubmit\\n';

for (i=0; i < data['ContactID'].length; i++) {
    var currRow = [data['ContactID'][i].toString(), data['FormSubmit'][i].toString().concat('\\n')];
    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'no_form_submit.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
""")
button2 = Button(label='Download', button_type='success', callback=callback2)

###############################################################################################################################


#Yes
data_t41 = dict(
        ContactID=dft4_1.ContactID,
        FormSubmit=dft4_1.FormSubmit,
    )
source_t41 = ColumnDataSource(data_t41)

columns_t41 = [
        TableColumn(field="ContactID", title="Contact Id"),
        TableColumn(field="FormSubmit", title="Form Submit"),
    ]
data_table_t41 = DataTable(name="All",source=source_t41, columns=columns_t41, width=250, height=480)


# Submit Download Button
callback3 = CustomJS(args=dict(source=source_t41), code="""
var data = source.data;
var filetext = 'ContactID,FormSubmit\\n';

for (i=0; i < data['ContactID'].length; i++) {
    var currRow = [data['ContactID'][i].toString(), data['FormSubmit'][i].toString().concat('\\n')];
    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'form_submit.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
""")
button3 = Button(label='Download', button_type='success', callback=callback3)

###############################################################################################################################



row_t41 = row(widgetbox(div40),widgetbox(div41),widgetbox(div42),widgetbox(div43))
row_t42 = row(column(row(controls, data_table_t4)),column(widgetbox(data_table_t40),widgetbox(button2,width = 200)),column(widgetbox(data_table_t41),widgetbox(button3,width = 200)))
#show(widgetbox(data_table))
col_t41 =column(row_t41,row_t42)

tab4 = Panel(child=col_t41, title="Output")
tabs4 = Tabs(tabs=[ tab4 ])

#show(tabs4)


curdoc().clear()


#reset_output()
#output_notebook()

fig1 = figure()
fig1.circle([0],[0])
tab_invsible = Panel(child=fig1, title='')

tabs5 = Tabs(tabs=[tab2,tab3,tab1,tab4,tab_invsible])

#tabs = Tabs(tabs=[tab1])

##output_file("Output.html")
##curstate().file['resources'] = MyResources(mode='cdn')
##save(tabs5)
#view("./Output.html")

curdoc().add_root(tabs5)
curdoc().title = "Sliders"
