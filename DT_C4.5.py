
# coding: utf-8

# 
# # Build a decision tree from scratch with c4.5 Algorithm
# 
# # Md Nasim Al Sajjad
# 

# In[43]:

import graphlab
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from collections import Counter
import matplotlib.patches as mpatches
get_ipython().magic(u'matplotlib inline')


# # Load training data

# In[2]:

df_train = graphlab.SFrame.read_csv('/Users/Sajjad/Downloads/dtt/decision_tree_training_data.txt',header=False)


# In[66]:

df_train.head(20)


# # Load Test Data

# In[3]:

df_test = graphlab.SFrame.read_csv('/Users/Sajjad/Downloads/dtt/decision_tree_test_data.txt',header=False)


# In[4]:

df_train
a =df_train.column_names()

df_train[a[0]]
df_train.head()
df_positive = df_train[df_train['X10']== 'positive']
df_negative = df_train[df_train['X10']== 'negative']
len(df_positive)
#len(df_negative)
eliment_count = dict(Counter(df_train[a[0]]))


# # Some utility Functions

# In[5]:

import math
def calculate_log2(x):
    return (math.log(x)/math.log(2))


# In[62]:


def count_entropy(column_name,df):
    eliment_count = dict(Counter(df[column_name]))
    o_cal = float(eliment_count['o'])/len(df)
    x_cal = float(eliment_count['x'])/len(df)
    b_cal = float(eliment_count['b'])/len(df)
    
    entropy = -(o_cal* calculate_log2(o_cal)) 
    - (x_cal*calculate_log2(x_cal))
    - (b_cal*calculate_log2(b_cal))
    return entropy


# In[7]:

def count_conditional_entropy(colum_name,df):
    df_o = df[df[colum_name]== 'o']
    df_x = df_train[df_train[colum_name]== 'x']
    df_b = df_train[df_train[colum_name]== 'b']
    E_o = count_Y_entropy(df_o)
    E_x = count_Y_entropy(df_x)
    E_b = count_Y_entropy(df_b)
    entropy = ((float(len(df_o)))/len(df))*E_o
    + ((float(len(df_x)))/len(df))*E_x + 
    ((float(len(df_b)))/len(df))*E_b
    return entropy


# # count_conditional_entropy

# In[8]:

def count_Y_entropy(df):
    eliment_count = dict(Counter(df['X10']))
    negative = float(eliment_count['negative'])/len(df)
    positive = float(eliment_count['positive'])/len(df)
    entropy = -(positive* calculate_log2(positive)) 
    - (negative*calculate_log2(negative))
    return entropy


# In[9]:

def calculate_information_gain(colum_name,df):
    x_entropy = count_entropy(colum_name,df)
    y_entropy = count_Y_entropy(df)
    x_y_entropy = count_conditional_entropy(colum_name,df)
    gain_x = (y_entropy-x_y_entropy)/x_entropy
    return gain_x
    


# calculate_information_gain(a[8],df_train)

# In[54]:

def find_IG(df):
    a =df.column_names()
    a = a[:len(a)-1]
    IG_list = []
    for i in (a):
        IG = calculate_information_gain(i,df_train)
        IG_list.append(IG)
    return IG_list


# In[55]:

ab=find_IG(df_train)
len(ab)


# In[56]:

def get_index_mac_IG(ig_list):
    max_value = max(ig_list)
    d_list = deque(ig_list)
    return max_value,list(d_list).index(max_value)
get_index_mac_IG(ab)


# In[57]:

#get_index_mac_IG(a)
cl = df_train.column_names()
xs =df_train[df_train[cl[4]]=='o']
xs


# In[16]:

def get_new_df(df, ind):
    new_df = df.remove_column(df.column_names()[ind])
    return new_df


# In[17]:

#get_new_df(df_train,4)


# In[63]:

def select_decision_feature(df):
    if (df is None):
        pass
    else:
        clm = df.column_names()
        if(len(clm)>1):
            ig =find_IG(df)
            max_value,idx = get_index_mac_IG(ig)
         
            print ('best feature is  '+ 
                   str(clm[idx])+ '  with ig  '+ str(max(ig)))
        else:
            return
        
    return max_value,idx,clm[idx]
                    


# In[18]:

gain,gain_idx, feature = select_decision_feature(df_train)


# In[19]:


def get_df_O(idx,df):
    clm = df.column_names()
    dfo = df[df[clm[idx]]=='o']
    dfo = get_new_df(dfo,idx)
    return dfo
    


# In[20]:

def get_df_X(idx,df):
    clm = df.column_names()
    dfx = df[df[clm[idx]]=='x']
    dfx = get_new_df(dfx,idx)
    return dfx
    


# In[21]:

def get_df_B(idx,df):
    clm = df.column_names()
    dfb = df[df[clm[idx]]=='b']
    dfb = get_new_df(dfb,idx)
    return dfb
    


# In[22]:

def get_decision_tree(df):
    gain,idx, feat = select_decision_feature(df)
    dfo = get_df_O(idx,df)
    dfx = get_df_X(idx,df)
    dfb = get_df_B(idx,df)
    return feat, dfo,dfx,dfb
           


# In[23]:

n = len(df_train.column_names())
tt = [[[graphlab.SFrame() for k in xrange(0,3)] for j in xrange(0,int(math.pow(3,i)))] for i in xrange(0,n)]


# In[24]:

n


# # Get all features

# In[25]:


ft = []
for i in range(0,10):
    if (i==0):
        if (df_train is None):
            pass
        else:
        
            feat,dfo,dfx,dfb =get_decision_tree(df_train)
            ft.append(feat)
       
            tt[i+1][0]=dfo.copy()
            tt[i+1][1]=dfx.copy()
            tt[i+1][2]=dfb.copy()
    else:
        level = int(math.pow(3,i))
        for j in range(0,level):
            k = j*3
            if(j%3 == 0):
                if (tt[i][j] is None):
                    pass
                else:
                
                    feat,dfo,dfx,dfb =get_decision_tree(tt[i][j])
                    ft.append(feat)
                    tt[i+1][k] = dfo.copy()
                    tt[i+1][k+1] = dfx.copy()
                    tt[i+1][k+2] = dfb.copy()
               
                
            elif (j%3 == 1):
                if (tt[i][j] is None):
                    pass
                else:
                    feat,dfo,dfx,dfb =get_decision_tree(tt[i][j])
                    ft.append(feat)
                    tt[i+1][k] = dfo.copy()
                    tt[i+1][k+1] = dfx.copy()
                    tt[i+1][k+2] = dfb.copy()
            elif (j%3 == 2):
                if (tt[i][j] is None):
                    pass
                else:
                    feat,dfo,dfx,dfb =get_decision_tree(tt[i][j])
                    ft.append(feat)
                    tt[i+1][k] = dfo.copy()
                    tt[i+1][k+1] = dfx.copy()
                    tt[i+1][k+2] = dfb.copy()


# In[26]:

ft
    


# # Get All paths

# In[27]:

p_0 = []

for i in range(0,len(ft)):
   
        
    fet = ft[i]
    
    locals()['p_{0}'.format(i*3+1)] = locals()['p_{0}'.format(i)][:]
    
    locals()['p_{0}'.format(i*3+1)].append({fet: 'o'})
    
    locals()['p_{0}'.format(i*3+2)] = locals()['p_{0}'.format(i)][:]
    
    locals()['p_{0}'.format(i*3+2)].append({fet: 'x'})
    
    
    locals()['p_{0}'.format(i*3+3)] = locals()['p_{0}'.format(i)][:]
    
    locals()['p_{0}'.format(i*3+3)].append({fet: 'b'})
        
    
       
       
        
    


# In[28]:


    
def convert_to_dict(p):
    new_dict = {}
    for i in range(0,len(p)):
        new_dict.update(p[i])
    return new_dict
    
    


# In[29]:

def error(df,rs):
    a = df['X10']
    b = rs
    x =0
    for i in range(0,len(df)):
        x +=(a[i] != b[i])
    return float(x)/len(df)


# # Make a copy of train set

# In[30]:

df = df_train.copy()
df = df.remove_column('X10')



# # Compute total possible paths

# In[31]:

def r(n):
    return 3*(math.pow(3,n-1))


# 

# In[32]:

def s(n):
    if (n==0):
        return 1
    return int(r(n)+s(n-1))


# In[ ]:




# # Get possible paths with top n features

# In[33]:

def get_path(n):
    patx = []
    for i in range(s(n),s(n+1)):
        d = convert_to_dict(globals()['p_{0}'.format(i)][:])
        patx.append(d)
    return patx
    


# # Learn with train data 
#     

# In[34]:

def get_tr_path(df,path):
    rs = [] 
    ps = []
    for i in range(0,len(df)):
        a=df[i]
    
        for j in range(0,len(path)):
            s = path[j]
            if(s.viewitems()<= a.viewitems()):
                res = df[i]['X10']
                p = path[j]
            
        rs.append(res)
        ps.append(p)
    return rs,ps
    


# # prediction path for test data

# In[35]:

def test_path(df,ps,rs):
    pr = []
    for i in range(0,len(df)):
        a=df[i]
    
        for j in range(0,len(ps)):
            s = ps[j]
            if(s.viewitems()<= a.viewitems()):
                pin = rs[j]
            
            
        pr.append(pin)
    return pr


# # Lets build the model

# In[36]:

def model(f,df_train,df_test):
    path = get_path(f)
    rs,ps = get_tr_path(df_train,path)
    pr = test_path(df_test,ps,rs)
    train_error = error(df_train,rs)
    test_error = error(df_test,pr)
    return train_error,test_error
    
    


# # Test the model with 4 top features

# In[39]:

a,b =model(4,df_train,df_test)
print(a)
print(b)


# # list error all posible models with # of features in between (2,6)

# In[40]:


x = []
y = []
for i in range (2,6):
    x.append(i+1)
    a,b = model(i,df_train,df_test)
    y.append(b)
    
    
    


# # Plot error vs # of input features

# In[65]:

red_patch = mpatches.Patch(color='blue', label='Error vs # Features')
plt.plot(x,y)
plt.legend(handles=[red_patch])
plt.xlabel('Number of features')
plt.ylabel('Percentage of Test Error')
plt.show()


# # So, when # of Features are 4, we get the lowest error

# # Lets Build a decision tree with Built In library

# In[50]:

fet = df_train.column_names()
tar = fet[-1]
fet = fet[:len(fet)-1]
decision_tree_model = graphlab.decision_tree_classifier.create(df_train,
                                                               validation_set=None,
                                target = tar, features = fet,max_depth =5)


# # Visualize the tree

# In[52]:

graphlab.canvas.set_target('ipynb')
decision_tree_model.show(view='Tree')

