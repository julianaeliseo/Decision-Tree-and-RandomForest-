import pandas as pd

df = pd.read_csv("C:/Users/julia/DataScience JN/creditcard.csv")

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()

#split dataset in features and target variable
feature_cols = ['V18','V16','V11','V10', 'V12', 'V14', 'V17']

X = df[feature_cols]  # Features
y = df.Class  # Labels

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
valor1 = 0.9995669627705019
dif  = (metrics.accuracy_score(y_test, y_pred))-valor1
print(valor1)
print(dif)