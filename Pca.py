#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[72]:


df = pd.read_csv("C://Users//win-10//Downloads//wdbc.data")


# In[73]:


col_names=['Id', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
             'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean','fractal dimension_mean','radius_se', 'texture_se', 'perimeter_se', 
             'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se','fractal dimension_se','radius_worst', 'texture_worst', 'perimeter_worst', 
             'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst','fractal dimension_worst']

df.columns=col_names

df.columns


# In[74]:


df_x = df.iloc[:,2:]
df_y = df.iloc[:,1]


# In[ ]:





# In[75]:


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


# In[76]:


final_df.describe()


# In[77]:


from sklearn.decomposition import PCA
pca=PCA(n_components=3)
principalComponents = pca.fit_transform(final_df)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1','PC2','PC3'])


# In[78]:


pd.set_option('display.max_columns', None)
principalDf


# In[79]:


finalDf = pd.concat([principalDf, df[['Diagnosis']]], axis = 1)


# In[80]:


finalDf.head()


# In[81]:


x_axis=finalDf['PC1']
y_axis=finalDf['PC2']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue =finalDf['Diagnosis'],palette=['g','r'] )
plt.title('Clusters by PCA Components')
plt.show()


# In[83]:


x_axis=finalDf['PC2']
y_axis=finalDf['PC3']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue =finalDf['Diagnosis'],palette=['g','r'] )
plt.title('Clusters by PCA Components')
plt.show()


# In[84]:


x_axis=finalDf['PC1']
y_axis=finalDf['PC3']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue =finalDf['Diagnosis'],palette=['g','r'] )
plt.title('Clusters by PCA Components')
plt.show()


# In[ ]:





# In[ ]:





# In[14]:


dfm=finalDf[finalDf['Diagnosis']=='M']
dfb=finalDf[finalDf['Diagnosis']=='B']


# In[15]:


col1=dfm['PC1']
col2=dfb['PC2']
plt.hist(col1, bins=20, alpha=0.5, label='M')
plt.hist(col2, bins=20, alpha=0.5, label='B')
plt.legend(loc='upper right')
plt.show()


# In[ ]:





# In[16]:


col1=dfm['PC1']
col2=dfb['PC3']
plt.hist(col1, bins=20, alpha=0.5, label='M')
plt.hist(col2, bins=20, alpha=0.5, label='B')
plt.legend(loc='upper right')
plt.show()


# In[17]:


col1=dfm['PC2']
col2=dfb['PC3']
plt.hist(col1, bins=20, alpha=0.5, label='M')
plt.hist(col2, bins=20, alpha=0.5, label='B')
plt.legend(loc='upper right')
plt.show()


# In[85]:


finalDf


# In[89]:


finalDf.to_excel(r'D:\File.xlsx')


# In[86]:


percentage_var_explained = pca.explained_variance_ratio_


# In[90]:


pca.components_.tolist()


# In[88]:


print("components:", pca.components_)
print("explained vaiance     ", pca.explained_variance_)
print("covariance:", pca.get_covariance()) 


# In[35]:


data=pca.explained_variance_
data


# In[36]:


plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], data)
plt.xlabel('PC#')
plt.ylabel('EigenValue')
plt.show()


# In[65]:


import matplotlib.pyplot as plt
handle=['EigenValues','Kaiser Rule']
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], data)
plt.axhline(y=1, linewidth=1.0,color='r')
plt.xlabel('PC#')
plt.ylabel('EigenValue')
plt.legend(handle,loc='upper left')
plt.show()


# In[48]:


data
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], data)

plt.xlabel('PC#')
plt.ylabel('EigenValue')
plt.show()


# In[49]:


data


# In[52]:


data.tolist()


# In[53]:


pca.components_.tolist().to


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




