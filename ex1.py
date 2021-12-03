import pandas as pd
import pylab as pl
import numpy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import copy
import math
import warnings
import matplotlib.cbook
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

df = pd.read_csv('healthcare-dataset-stroke-data.csv') #read the data

x = df.describe() #get data's statistic properties
print(x)

#Ages histogram
plt.subplot(121)
df[0:250].age.plot.hist(grid=True, bins=100, rwidth=0.9)
plt.subplot(122)
df[250:-1].age.plot.hist(grid=True, bins=100, rwidth=0.9)

#Stroke depending on other features
df.hist(sharex='True' ,column="age", by ="stroke")
pl.suptitle("Strokes depending on age")

df.hist(column="bmi", by="stroke")
pl.suptitle("Strokes depending on bmi")

df.hist(column="avg_glucose_level", by="stroke")
pl.suptitle("Strokes depending on average glucose level")

df.hist(column="hypertension", by="stroke")
pl.suptitle("Strokes depending on hypertension")

df.hist(column="heart_disease", by="stroke")
pl.suptitle("Strokes depending on heart disease")

plt.show() #need to close them so that the program continues

[rows,columns]=df.shape

##################################################
#       Drop features with missing values
##################################################
sample_data=copy.copy(df)
mod_data=sample_data.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
A = mod_data.drop([col for col in mod_data.columns if mod_data[col].eq('Unknown').any()], axis=1)

##################################################
#               Mean Value
##################################################
#numeric missing values df.isna().sum()
B = copy.copy(df)
B.drop('id', inplace = True, axis =1)
for i in range(0,rows): # replace missing value with mean value
    if numpy.isnan(B.bmi[i]) == True:
        B.bmi[i] = x.bmi[1]
    else:
        continue
B.drop('smoking_status', inplace =True, axis=1)

##################################################
#               Linear Regression
##################################################

C=copy.copy(df)
C.drop('smoking_status', inplace =True, axis=1) #drop smoking_status column because it has categorical data
C.drop('id', inplace = True, axis =1) #drop id because it has nothing to do with the classification process

C2=copy.copy(C)
le_gender=LabelEncoder() #Catergorical Values Encoding so that they become numerical
C2 = C2.assign(gender = le_gender.fit_transform(C2.gender))
le_married=LabelEncoder()
C2 = C2.assign(ever_married = le_married.fit_transform(C2.ever_married))
le_work=LabelEncoder()
C2 = C2.assign(work_type = le_work.fit_transform(C2.work_type))
le_residence=LabelEncoder()
C2 = C2.assign(Residence_type = le_residence.fit_transform(C2.Residence_type))

min_max_scaler = MinMaxScaler() #Normalize all data for more reliable results
C2[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","stroke"]] = min_max_scaler.fit_transform(C2[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","stroke"]])

col_nan_ix = C2[C2['bmi'].isnull()].index #rows where nan values exist

known = C2.dropna(inplace=False)

known_feat = known.drop('bmi', inplace = False,axis =1)
known_label = known.bmi

#Determine two features with the highest correlation to bmi
print(known.corr(method = 'pearson')['bmi'])

frame={'Ever Married': known.ever_married, 'Age': known.age}
features=pd.DataFrame(frame)

label = pd.DataFrame(known.bmi)

# Train Linear Regression Model
lr=LinearRegression()
lr.fit(features,label)

r_sq = lr.score(features,label)
print('coefficient of determination:', r_sq)

C3 = C2[C2.index.isin(col_nan_ix)]

frame={'Ever Married': C3.ever_married, 'Age': C3.age}
x_predict=pd.DataFrame(frame)

bmi_predict = lr.predict(x_predict)

k=0
for i in col_nan_ix:
    C2.loc[int(i),'bmi'] = bmi_predict[k]
    k=k+1
    
#df.loc[col_nan_ix,'bmi'] = bmi_predict

#inverse transform 
o=min_max_scaler.inverse_transform(C2)
u=pd.DataFrame(o)
C.bmi = u.iloc[:,8]



##################################################
#                   KNN
##################################################

D=copy.copy(df)
D.drop('bmi', inplace =True, axis=1) #drop bmi column because it has numerical data
D.drop('id', inplace = True, axis =1) #drop id because it has nothing to do with the classification process

D2=copy.copy(D)

le_gender=LabelEncoder() #Categorical Values Encoding
D2 = D2.assign(gender = le_gender.fit_transform(D2.gender))

le_married=LabelEncoder()
D2 = D2.assign(ever_married = le_married.fit_transform(D2.ever_married))

le_work=LabelEncoder()
D2 = D2.assign(work_type = le_work.fit_transform(D2.work_type))

le_residence=LabelEncoder()
D2 = D2.assign(Residence_type = le_residence.fit_transform(D2.Residence_type))

le_smoking=LabelEncoder()
D2 = D2.assign(smoking_status = le_smoking.fit_transform(D2.smoking_status))

min_max_scaler = MinMaxScaler() #Normalize all data for more reliable kNN classification
D2[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","stroke"]] = min_max_scaler.fit_transform(D2[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","stroke"]])

unknown = D2.loc[D2['smoking_status'] == 0]
known = D2.loc[D2['smoking_status'] != 0]

known_feat = known.drop('smoking_status', inplace = False,axis =1)
known_label = known.smoking_status

unknown_indexes = df[df['smoking_status']=='Unknown'].index.values
smoking_predict = D2[D2.index.isin(unknown_indexes)]
smoking_predict.drop('smoking_status', inplace = True, axis=1)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(known_feat, known_label, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=71)

knn.fit(X_train, y_train)

#Predict test dataset
y_pred = knn.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Predict Unknown values 
status_pred = knn.predict(smoking_predict)

#Replace Unknown values in initial dataframe with KNN predicted values
k=0
for i in unknown_indexes:
    D2.loc[int(i),'smoking_status'] = status_pred[k]
    k=k+1

ss=D2.smoking_status
v = le_smoking.inverse_transform(ss)
s = pd.DataFrame(v)
D.smoking_status = s

##################################################
#           Linear Regression + KNN
##################################################

E = copy.copy(df)
E.drop('id', inplace = True, axis =1)
E.bmi = C.bmi
E.smoking_status = D.smoking_status

##################################################
#           Random Forest
##################################################

def RF(M):

    le_gender=LabelEncoder() #Categorical Values Encoding
    M = M.assign(gender = le_gender.fit_transform(M.gender))

    le_married=LabelEncoder()
    M = M.assign(ever_married = le_married.fit_transform(M.ever_married))

    le_work=LabelEncoder()
    M = M.assign(work_type = le_work.fit_transform(M.work_type))

    le_residence=LabelEncoder()
    M = M.assign(Residence_type = le_residence.fit_transform(M.Residence_type))
    
    if 'smoking_status' in M.columns:
        le_smoking=LabelEncoder()
        M = M.assign(smoking_status = le_smoking.fit_transform(M.smoking_status))
    
    if ('smoking_status' not in M.columns) and ('bmi' not in M.columns):
        min_max_scaler = MinMaxScaler() #Normalize all data for more reliable kNN classification
        M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level"]] = min_max_scaler.fit_transform(M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level"]])
    
    elif 'smoking_status' not in M.columns:
        min_max_scaler = MinMaxScaler() #Normalize all data for more reliable kNN classification
        M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi"]] = min_max_scaler.fit_transform(M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi"]])

    elif 'bmi' not in M.columns:
        min_max_scaler = MinMaxScaler() #Normalize all data for more reliable kNN classification
        M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","smoking_status"]] = min_max_scaler.fit_transform(M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","smoking_status"]])

    elif ('smoking_status' in M.columns) and ('bmi' in M.columns) :
        min_max_scaler = MinMaxScaler() #Normalize all data for more reliable kNN classification
        M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]] = min_max_scaler.fit_transform(M[["gender", "age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]])
    
    stroke_class = M.stroke
    X_train, X_test, y_train, y_test = train_test_split(M.drop(['stroke'],axis='columns'), stroke_class, test_size=0.25, random_state=100)
    model = RandomForestClassifier(n_estimators = 200, random_state = 100)
    model.fit(X_train,y_train)
    accuracy = model.score(X_test,y_test)
    q = model.predict(X_test)

    metric = precision_recall_fscore_support(y_test, q, average='macro')
    print("\nPrecision: " + str(metric[0]))
    print("Recall: " + str(metric[1]))
    print("F1-Score: " + str(metric[2]))
    cm=confusion_matrix(y_test,q)
    print("\n\nConfusion Matrix:\n" + str(cm))


print('Case A: Drop features with missing values')
RF(A)
print("###################\n")
print('Case B: Replace missing with mean value')

RF(B)
print("###################\n")
print('Case C: Predict missing values with linear regression')

RF(C)
print("###################\n")
print('Case D: Predict missing values with KNN')
RF(D)

print("###################\n")
print('Case E: Predict numerical values with linear regression and KNN')
RF(E)

