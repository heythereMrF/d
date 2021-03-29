#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[3]:


df = pd.read_csv("C://Users//win-10//Downloads//wdbc.data")


# In[4]:


col_names=['Id', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
             'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean','fractal dimension_mean','radius_se', 'texture_se', 'perimeter_se', 
             'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se','fractal dimension_se','radius_worst', 'texture_worst', 'perimeter_worst', 
             'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst','fractal dimension_worst']

df.columns=col_names

df.columns


# In[5]:


df_x = df.iloc[:,2:]
df_y = df.iloc[:,1]


# In[ ]:





# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_x)
scaled_df = scaler.transform(df_x.values)
final_df = pd.DataFrame(scaled_df,columns=['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal dimension_worst'])


# In[7]:


final_df


# In[8]:


final_df.describe()


# In[10]:


from sklearn.decomposition import PCA
pca=PCA(n_components=3)
principalComponents = pca.fit_transform(final_df)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1','PC2','PC3'])


# In[11]:


pd.set_option('display.max_columns', None)
principalDf


# In[12]:


finalDf = pd.concat([principalDf, df[['Diagnosis']]], axis = 1)


# In[ ]:





# In[13]:


from sklearn.cluster import KMeans


# In[389]:


kmeans_pca = KMeans(n_clusters=2, init='random', max_iter=400, n_init=200, random_state=0)
kmeans_pca.fit(principalDf)


# In[390]:


Principal_kmeans = pd.concat([df_x.reset_index(drop=True),pd.DataFrame(principalDf)],axis=1)
Principal_kmeans.columns.values[-3: ]= ['PC1','PC2','PC3']


# In[391]:


kmeans_pca.cluster_centers_.tolist()


# In[392]:


from sklearn.metrics import davies_bouldin_score


# In[394]:


labels = kmeans_pca.labels_
davies_bouldin_score(principalDf, labels)


# In[ ]:





# In[395]:


pk=kmeans_pca.cluster_centers_
pk
pk2=pk[0]
pk2.tolist()


# In[396]:


Principal_kmeans['Segment Kmeans values']= kmeans_pca.labels_


# In[397]:


Principal_kmeans.head()


# In[398]:


Final_Principle= pd.concat([Principal_kmeans, df[['Diagnosis']]], axis = 1)


# In[400]:


x_axis=Principal_kmeans['PC1']
y_axis=Principal_kmeans['PC2']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue =Principal_kmeans['Segment Kmeans values'],palette=['g','r'] )
plt.title('Clusters by PCA Components')
plt.show()


# In[ ]:





# In[ ]:





# In[401]:


Final_Principle.head()


# In[402]:


kmeans_pca.cluster_centers_


# In[403]:


#Final_Principle[(Final_Principle['Segment Kmeans values']==0)&(Final_Principle['Diagnosis']=='M')].count()
#df[(df['A']>0) & (df['B']>0) & (df['C']>0)].count()


# In[ ]:





# In[404]:


from sklearn.cluster import KMeans


# In[405]:


kmeans_pca = KMeans(n_clusters=3,init='k-means++',random_state=20)
kmeans_pca.fit(principalDf)


# In[406]:


Principal_kmeans = pd.concat([df_x.reset_index(drop=True),pd.DataFrame(principalDf)],axis=1)
Principal_kmeans.columns.values[-30: ]= ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29','PC30']


# In[407]:


Principal_kmeans['Segment Kmeans values']= kmeans_pca.labels_


# In[408]:


Principal_kmeans.head()


# In[409]:


x_axis=Principal_kmeans['PC1']
y_axis=Principal_kmeans['PC2']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue =Principal_kmeans['Segment Kmeans values'],palette=['g','r','m'] )
plt.title('Clusters by PCA Components')
plt.show()


# In[410]:


Final_Principle2= pd.concat([Principal_kmeans, df[['Diagnosis']]], axis = 1)


# In[411]:


Final_Principle2.head()


# In[412]:


Final_Principle2[(Final_Principle2['Segment Kmeans values']==2)&(Final_Principle2['Diagnosis']=='M')].count()


# In[413]:


kmeans_pca.cluster_centers_


# In[414]:


from sklearn.metrics import davies_bouldin_score


# In[415]:


labels = kmeans_pca.labels_
davies_bouldin_score(principalDf, labels)


# In[416]:


from sklearn.cluster import KMeans


# In[417]:


kmeans_pca = KMeans(n_clusters=5,init='k-means++',random_state=20)
kmeans_pca.fit(principalDf)


# In[418]:


Principal_kmeans = pd.concat([df_x.reset_index(drop=True),pd.DataFrame(principalDf)],axis=1)
Principal_kmeans.columns.values[-30: ]= ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29','PC30']


# In[419]:


Principal_kmeans['Segment Kmeans values']= kmeans_pca.labels_


# In[420]:


Principal_kmeans.head()


# In[423]:


x_axis=Principal_kmeans['PC1']
y_axis=Principal_kmeans['PC2']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue =Principal_kmeans['Segment Kmeans values'],palette=['g','r','c','m','b'] )
plt.title('Clusters by PCA Components')
plt.show()


# In[424]:


Final_Principle3= pd.concat([Principal_kmeans, df[['Diagnosis']]], axis = 1)


# In[425]:


Final_Principle3.head()


# In[426]:


Final_Principle3[(Final_Principle3['Segment Kmeans values']==4)&(Final_Principle3['Diagnosis']=='B')].count()


# In[ ]:





# In[427]:


kmeans_pca.cluster_centers_


# In[428]:


from sklearn.metrics import davies_bouldin_score


# In[429]:


labels = kmeans_pca.labels_
davies_bouldin_score(principalDf, labels)


# In[ ]:




