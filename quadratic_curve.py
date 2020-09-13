import pandas as pd
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns

df = pd.read_csv('/content/sample_data/covid.csv')
#reading data
df.head()
#collecting data
s_no=""
x_date=""
new_d=""
i=0;
for col in df:
    if i==0:
        s_no=col
    if i==1:
        x_date=col
    if i==7:
        new_d=col
    i=i+1
s_n=df[s_no].values
x_dat=df[x_date].values
nd=df[new_d].values
l=s_n[91:113]
Y=nd[91:113]
X=[]
a=l[0]
print('plotting data from 2 may to 23 may')
for x in l:
    x=x-a
    X.append(x+1)
plt.scatter(X,Y)
plt.title("day vs death: Scatter Plot")
plt.ylabel("Number of death")
plt.xlabel("Day_no")
X=np.array(X)
X_mat=np.vstack(((np.ones(len(X)), X),X*X)).T
beta_hat = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y)
y_hat = X_mat.dot(beta_hat)
print(beta_hat)
plt.plot(X,y_hat,color='red')
#data prediction
Y_actual=nd[112:121]
X_data=[]
for i in range(9):
  X_data.append(22+i+1)
X_data=np.array(X_data)
X_ma=np.vstack(((np.ones(len(X_data)), X_data),X_data*X_data)).T
y_pred=X_ma.dot(beta_hat)
a=len(y_pred)
print('testing data')
print('date','actual_value  ','predicated_value')
error_sum=0
for i in range(a):
  print(i+22+1,'may 2020','    ',Y_actual[i],'     ',y_pred[i])
  h=abs(Y_actual[i]-y_pred[i])
  h=h*h
  error_sum+=h;
error_sum/=18
print('Mean Squared error in the test data from 23 may to 30 may is')
print(error_sum)
#data predication on april 20 and june 10
print('Value as predicted on april 20 and June 10 is')
data=[-10,41]
data=np.array(data)
X_m=np.vstack(((np.ones(len(data)),data),data*data)).T
Y_p=X_m.dot(beta_hat)
print('on april 20 :',Y_p[0])
print('on june 10:',Y_p[1])


