#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import re
import os

import json
import pickle
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

os.getcwd()


# ## Important faults: 55\- AC CONTACTOR  67\-BUS DIFF VOL 95\-DC\-FUSE
# 
# 

# ### Read data from module\_RR.pkl and use sliding window to generate train and test data
# 
# 

# read data

# In[23]:


path = '../Multilabel_classification/Data/module_RR.pkl'
file_name = path
f = open(file_name, 'rb')
data = pickle.load(f)
print(len(data))
X = data['data']
print('X shape:',X.shape)
Y = data['label']
print('Y shape:',Y.shape)
README = data['README']
print(README)


# sliding window

# ### 2.1 Faults classes check
# 
# 

# In[24]:


def fault_check(labels):
    unique_fault, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({'event':unique_fault, 'count':counts}).drop(0).sort_values(by=['count'])
    print("*******label and counts of original Y")
    print(unique_fault, counts)
    return df

p = '../Multilabel_classification/Data'
df_counts = fault_check(Y)
plt.figure(figsize=(8, 3), dpi=300)
df_counts["event"] = df_counts["event"].astype(int).astype(str)
df_counts.to_csv(Path(p,'countsRR_TS.csv'),index = False)
plt.bar(df_counts['event'].to_list(), df_counts['count'].to_list(), color ='maroon',
        width = 0.4)
plt.title('Counts of All Faults in the System')
plt.xlabel("Fault Type")
plt.ylabel("Count")
plt.savefig(Path(p,'countsRR_TS.png'), bbox_inches='tight')
plt.show()


# In[25]:


from sklearn import preprocessing

unique_fault, counts = np.unique(Y, return_counts=True)
print("*******label and counts of original Y")
unique_fault = unique_fault.tolist()
print(unique_fault, counts)
#only keep three faults
useful_code = [0.0, 55.0, 67.0, 95.0]
rare_list = [x for x in unique_fault if x not in useful_code]
rare_list = np.array(rare_list)
# rare_list = unique_fault[counts<8000]   # 9 for 8000; 6 for 20000
print('....',rare_list.shape)
precess_Y = Y.copy()
for rare_value in rare_list:
    precess_Y[precess_Y == rare_value] = 100   #assign another value to the 100
unique_fault1, counts1 = np.unique(precess_Y, return_counts=True)
print("*******label and counts after filter")
print(unique_fault1, counts1)

le = preprocessing.LabelEncoder()
shape_Y = precess_Y.shape
Y_le = le.fit_transform(precess_Y.flatten('C'))
Y_le = Y_le.reshape(shape_Y,order='C')   #Y lable after label embeding
Yshape = Y_le.shape
Y_reshape = Y_le.reshape((Yshape[0]*Yshape[1],Yshape[2]))
Y_reshape = np.swapaxes(Y_reshape,1,0)


# In[26]:


Xshape = X.shape
X_reshape = X.reshape((Xshape[0]*Xshape[1],Xshape[2],Xshape[3]))
X_reshape = np.swapaxes(X_reshape,2,0)
X_reshape = np.swapaxes(X_reshape,2,1)[...,:-1]


# In[29]:


print(X_reshape.shape,Y_reshape.shape)


# 

# ### 2.2 Sliding window and train\-test\-split
# 
# 

# In[30]:


stride=30 #timestamps
sequence_length=100 #timestamps
forecast_length=100  #timestamps

def sliding_windows(X,Y,stride,sequence_length,forecast_length):   #x and y have the same time steps
    x = []
    y = []
    forecast_y=[]
    for i in range(int((len(X)-sequence_length)/stride)+1-stride): 
        _x = X[i*stride:i*stride+sequence_length,...]
        _y = Y[i*stride:i*stride+sequence_length,...]
        if i<=(int((len(X)-sequence_length)/stride)+1)-stride-1:
            _f = Y[i*stride+sequence_length:i*stride+sequence_length+forecast_length,...]
            forecast_y.append(_f)
        x.append(_x)
        y.append(_y)
    return np.array(x),np.array(y),np.array(forecast_y)


Min_max   = MinMaxScaler()
X_norm    = X_reshape.reshape(-1,1)
X_norm    = Min_max.fit_transform(X_norm)
X_norm    = X_norm.reshape(X_reshape.shape)
train_len = int(np.floor(len(Y_reshape)/10*0.8)*10)-60
test_len  = int(np.floor(len(Y_reshape)/10*0.2)*10)-40
train_X   = X_norm[:train_len,:]
train_Y   = Y_reshape[:train_len]
test_X    = X_norm[train_len:train_len+test_len,:]
test_Y    = Y_reshape[train_len:train_len+test_len]


train_X_win, train_Y_win, train_f_y = sliding_windows(train_X,train_Y,stride,sequence_length,forecast_length)
test_X_win , test_Y_win , test_f_y  = sliding_windows(test_X,test_Y,stride,sequence_length,forecast_length)
print('Shape of traning X, Y forecast:',train_X_win.shape,train_Y_win.shape,train_f_y.shape)
print('Shape of test X, Y forecast:',   test_X_win.shape,test_Y_win.shape,test_f_y.shape)


# Put all inverter and modules data in the first dimension

# In[31]:


def All_Module_to_One(X,Y): 
    X      = np.swapaxes(X,1,2)
    Xshape = X.shape
    X      = X.reshape((Xshape[0]*Xshape[1],Xshape[2],Xshape[3]))
    Y      = np.swapaxes(Y,1,2)
    Yshape = Y.shape
    Y      = Y.reshape((Yshape[0]*Yshape[1],Yshape[2]))
    print('shapes of X and Y:',X.shape,Y.shape)
    return X, Y

train_X, train_Y = All_Module_to_One(train_X_win, train_Y_win)
test_X ,test_Y   = All_Module_to_One(test_X_win, test_Y_win)




# In[32]:


#convert y labels to multip label

def multievent(labels): 
    one_hot      = MultiLabelBinarizer()
    y_multilabel = one_hot.fit_transform(labels)
    y_multi      = y_multilabel[:,1:]  #Only keep only fault columns
    print('Size of new generated labels:',y_multi.shape)
    return y_multi

train_multi = multievent(train_Y)
test_multi  = multievent(test_Y)
    # print('If there is any multi-labels:',2 in np.sum(y_multilabel,axis=1))


# In[33]:


summation  = test_multi.sum(1)
summation_train = train_multi.sum(1)
num_nonzero = list(summation_train).count(0)
num_nonzero = len(summation_train) - num_nonzero
print(num_nonzero)  #
num_nonzero = list(summation).count(0)
num_nonzero = len(summation) - num_nonzero
print(num_nonzero)  #


# In[34]:


print(np.unique(train_Y))
a = np.array([ 0,  1,  2,  3,4])
reverse_label = le.inverse_transform(a)
print(reverse_label)


# ## set device and torchdataset

# In[35]:


import torch.utils.data as data_utils
device='cuda:3'
BS=40
train_data=data_utils.TensorDataset(torch.Tensor(train_X).float().to(device),torch.Tensor(train_multi).float().to(device))
train_data_loader = torch.utils.data.DataLoader(train_data,batch_size =BS,shuffle=True)
test_data=data_utils.TensorDataset(torch.Tensor(test_X).float().to(device),torch.Tensor(test_multi).float().to(device))
test_data_loader = torch.utils.data.DataLoader(test_data,batch_size =10000)


# # CNN
# 
# 

# In[79]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=64,    # n_filters
                kernel_size=(5,2),      # filter size
                stride=2,           # filter movement/step
                padding=1,
            ),     
            nn.MaxPool2d(kernel_size=2),   
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(64, 32, (5,2), 1, 1),  
#             nn.ReLU(),  # activation
            nn.MaxPool2d(2), 
        )
        self.out = nn.Linear(352 , 4)
#         self.out2 = nn.Linear(32 , 4)   # fully connected layer, output 5 classes

    def forward(self, x):
        x = self.conv1(x)
        # print('1:',x.shape)
        x = self.conv2(x)
        # print('2:',x.shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        output = self.out(x)
        return output



# In[80]:


EPOCH     = 150
cnn       = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0002)  
loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]).to(device))

for epoch in range(EPOCH): 
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.05/(epoch+1))  
    train_loss = []
    for step, (b_x, b_y) in enumerate(train_data_loader):
        output = cnn(b_x.unsqueeze(1))                  # cnn output
        loss   = loss_func(output, b_y)                 # loss
        train_loss.append(loss)
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
    avg_loss = sum(train_loss)/len(train_data_loader)

    print('Epoch: ',epoch,' loss: ',avg_loss)


# ### save trained model
# 
# 

# In[81]:


save_path = '../Multilabel_classification/saved_models/rr_imp_detection_556795_w5.pth'
torch.save(cnn.state_dict(), save_path)


# In[84]:


loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]).to(device))

path = '../Multilabel_classification/saved_models/rr_imp_detection_556795_w5.pth'
cnn = CNN()
cnn.load_state_dict(torch.load(path))
cnn.eval()
cnn.to(device)


# ### Define test
# 
# 

# In[85]:


batch_size = 10000
def test(net,testloader):

    test_loss=0.0
    test_results =torch.zeros_like(torch.tensor(test_multi))
    net.eval()
    test_loss=0.0
    for i, (data, target) in enumerate(testloader):
        print(data.shape)
        output = net(data.unsqueeze(1))
#         print(output.shape)
        # calculate the loss
        loss = loss_func(output, target)
        # update running validation loss
        test_loss += loss.item()*data.size(0)
        if data.size(0)==batch_size:
            test_results[batch_size*i:batch_size*(i+1)]=output.squeeze()
        else:
            test_results[batch_size*i:]=output.squeeze()
    test_loss_avg = test_loss/len(testloader.sampler)
    print('loss: ',test_loss_avg)
    return test_loss_avg, test_results


# In[86]:


test_loss, prediction = test(cnn,test_data_loader)


# In[87]:


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[88]:


from sklearn.metrics import f1_score
import seaborn as sns
print(prediction.shape)
probs=torch.sigmoid(prediction)
binary_matrix= probs[:,:]>0.5
from sklearn.metrics import confusion_matrix
probs_np=binary_matrix.cpu().numpy()
test_y_np=test_multi
cf_matrix=[]
for i in range(4):
    f1 = f1_score(test_y_np[:,i], binary_matrix[:,i])
    cf = confusion_matrix(test_y_np[:,i], probs_np[:,i])
    cf_matrix.append(cf)
    print(cf)
    print('F1 score:',f1)
    sns.heatmap(cf_matrix[i], annot=True)


# In[89]:


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
Path_folder = '../Multilabel_classification/result_img/report/fault_3_5_oneday'
create_directory(Path_folder)
for i in range(4):
    f, ax = plt.subplots(figsize=(4, 3),dpi = 180)
    plt.tight_layout()
    sns.heatmap(cf_matrix[i],  annot=True,fmt=".0f")
    plt.savefig(Path(Path_folder,str(i)+'.png'))
    plt.show()


# ## Overall F1 and Confusion matrix

# In[90]:


binary = np.mean(test_multi,axis=1)>0
binary.shape
pred_binary = np.mean(np.array(binary_matrix),axis=1)>0
print(pred_binary.shape)


# In[91]:


cf_detection = confusion_matrix(binary, pred_binary)
cf_detection
plt.figure(figsize=(4, 3),dpi = 220)
f1_detection = f1_score(binary, pred_binary)
sns.heatmap(cf_detection,  annot=True,fmt=".0f")
plt.savefig(Path(Path_folder,'overall_detection.png'))
plt.show()
print(f1_detection)


# In[ ]:




