import pandas as pd
from sklearn import preprocessing
from sklearn import tree
data=pd.read_csv('cpdata.csv')

inputs=data.drop('label',axis='columns')
print(inputs)

label_encoder = preprocessing.LabelEncoder()
data['label']= label_encoder.fit_transform(data['label'])

target=data['label']
targetarray=data['label'].unique()
print(targetarray)
print(target)

model=tree.DecisionTreeClassifier()
model.fit(inputs,target)
print(model.score(inputs,target))
index=0
k=model.predict([[22.71,54.72,6.90,146.66]])
print(k)
for i in range(0,31):
    if(k==targetarray[i]):
        index=i
        break


crops=['rice','wheat','Mung Bean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','pomegranate','watermelon']
print(crops[index])