import pandas as pd
df = pd.read_csv("C:/Users/julia/DataScience JN/creditcard.csv")
print(df)

import os
import matplotlib.pyplot as plt

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# load dataset
#split dataset in features and target variable
feature_cols = ['Time','V1','V2', 'V3', 'V4', 'V5', 'V6','V7','V8','V9', 'V10',
                'V11', 'V12','V13','V14','V15','V16','V17','V18','V19', 'V20',
                'V21', 'V22', 'V23', 'V24','V25','V26','V27','V28']
X = df[feature_cols] # Features
y = df.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('creditcardfraud.png')
Image(graph.create_png())

import cv2

gray = cv2.imread('creditcardfraud.png', 0)
plt.figure(figsize=(12,8))
plt.imshow(gray)
plt.show()
