import pandas as pd

df = pd.read_csv("C:/Users/julia/DataScience JN/creditcard.csv")

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()

#split dataset in features and target variable
feature_cols = ['Time','V1','V2', 'V3', 'V4', 'V5', 'V6','V7','V8','V9', 'V10',
                'V11', 'V12','V13','V14','V15','V16','V17','V18','V19', 'V20',
                'V21', 'V22', 'V23', 'V24','V25','V26','V27','V28']

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
valor1 = metrics.accuracy_score(y_test, y_pred)
#print(clf.predict([[1,0.3,0.2, 0.1, -0.02, -0.3, 0.4, 1.45, 0.8,0.9, 0.78,
 #               2, 3,4, 1, 1,-1,-0.5,-0.4,-0.6, 0.88,
  #              0.4, 0.6, 0.5, 0.2, 0.3,1, -1, 0.8]]))

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index= feature_cols).sort_values(ascending=False)
print(feature_imp)

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Import train_test_split function

from sklearn.model_selection import train_test_split

newfeature_cols = ['Time','V1','V2', 'V3', 'V4', 'V5', 'V6','V7','V8','V9', 'V10',
                'V11', 'V12','V13','V14','V15','V16','V17','V18','V19', 'V20',
                'V21', 'V22','V26','V27','V28']

X=newfeature_cols  # Removed feature V24 V25 V23
y=df.Class
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)
# 70% training and 30% test



#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation

# Model Accuracy, how often is the classifier correct?
print(valor1)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


