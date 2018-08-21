
# coding: utf-8

# In[2]:


import os
import pandas as pd
import seaborn as sns
import time
sns.set_style('whitegrid')
path = 'G:\\Python\\Astronomy\\first_train_data_20180131'
dirs = os.listdir(path)
data = pd.read_csv('G:\\Python\\Astronomy\\first_train_index_20180131.csv', encoding = 'gbk')
import matplotlib.pyplot as plt
o = 1
while o <= 5:
    title = pd.read_table('G:\\Python\\Astronomy\\first_train_data_20180131\\' + dirs[o-1], sep=',',header = None)
    title = title.T
    title1 = []
    i = len(title)
    title = title.T
    k = 1
    o = o+1
    while k < i:
        if k % 1==0:
            title1.append(title.iloc[0,k])
        k = k + 1
    title1 = pd.DataFrame(title1)
    title1.plot()
    plt.title(data['type'][o-2])


# In[3]:


print (len(dirs))


# In[4]:

data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[ ]:


data['type'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split
root_path = "G:\\Python\\Astronomy\\first_train_data_20180131"
id_set = []
feat_set = []
path_set = []
def get_str_split(value):
    if len(value.split("."))>2:
        return value.split(".")[-2]+value.split(".")[-1]
    else:
        return value
for id_per in (data['id'].tolist()[:483851]):
    id_set.append(id_per)  #设置容器，存放数据
    dt = pd.read_table(('%s/%s.txt') % (root_path, id_per), sep=",") #读取文本数据
    dt_values = [get_str_split(col) for col in dt.columns.tolist()]  #检查数据
    feat_set.append(dt_values)
    # print feat_set

x_train = pd.DataFrame(feat_set)
y_train = data.iloc[:500, :]['type'].values.tolist()
for i in range(len(y_train)):
    if y_train[i] == 'star': y_train[i] = 0
    if y_train[i] == 'qso': y_train[i] = 1
    if y_train[i] == 'galaxy': y_train[i] = 2
    if y_train[i] == 'unknown': y_train[i] = 3
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
training_start = time.perf_counter()
knn.fit(x_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = knn.predict(x_test)
prediction_end = time.perf_counter()
acc_knn = (preds == y_test).sum().astype(float) / len(preds)*100
knn_train_time = training_end-training_start
knn_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's K Nearest Neighbors Classifier's prediction accuracy is: %3.2f" % (acc_knn))
print("Time consumed for training: %4.3f seconds" % (knn_train_time))
print("Time consumed for prediction: %6.5f seconds" % (knn_prediction_time))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
training_start = time.perf_counter()
gnb.fit(x_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = gnb.predict(x_test)
prediction_end = time.perf_counter()
acc_gnb = (preds == y_test).sum().astype(float) / len(preds)*100
gnb_train_time = training_end-training_start
gnb_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's Gaussian Naive Bayes Classifier's prediction accuracy is: %3.2f" % (acc_gnb))
print("Time consumed for training: %4.3f seconds" % (gnb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (gnb_prediction_time))


# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100)
training_start = time.perf_counter()
xgb.fit(x_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(x_test)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))


# In[ ]:


from sklearn.svm import SVC
svc = SVC()
training_start = time.perf_counter()
svc.fit(x_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = svc.predict(x_test)
prediction_end = time.perf_counter()
acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100
svc_train_time = training_end-training_start
svc_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's Support Vector Machine Classifier's prediction accuracy is: %3.2f" % (acc_svc))
print("Time consumed for training: %4.3f seconds" % (svc_train_time))
print("Time consumed for prediction: %6.5f seconds" % (svc_prediction_time))
print(preds)


# In[ ]:


results = pd.DataFrame({
    'Model': ['KNN', 'Naive Bayes', 
              'XGBoost', 'SVC'],
    'Score': [acc_knn, acc_gnb, acc_xgb,  acc_svc,],
    'Runtime Training': [knn_train_time, gnb_train_time, xgb_train_time,  
                         svc_train_time],
    'Runtime Prediction': [knn_prediction_time, gnb_prediction_time, xgb_prediction_time,
                          svc_prediction_time]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Model')
result_df

