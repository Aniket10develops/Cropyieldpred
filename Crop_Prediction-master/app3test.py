# Import label encoder
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
data=pd.read_csv('cpdata.csv')


# Encode labels in column 'species'.
data['label']= label_encoder.fit_transform(data['label'])
x=data.iloc[:, 0:4].values
y=data.iloc[:, 4:].values

print(x)
print(y)
print(data['label'].unique())
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
  
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# Finding the accuracy of the model
a=accuracy_score(y_test,dtree_predictions)

print(a)
ah=0
atemp=0
##shum=82.002744
pH=0
rain=0

l=[]

l.append(atemp)
l.append(ah)
l.append(pH)
l.append(rain)
predictcrop=[l]

predictions = dtree_model.predict(predictcrop)

print(predictions[0])
