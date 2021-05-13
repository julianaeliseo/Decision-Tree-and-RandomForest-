# Decision-Tree-and-RandomForest-
This project's aim is to acquire information about these two classifiers Decision Tree and Random Forest. 
Is there a significant gain between the models? The dataset is about credit card fraud.
Available on kaggle https://www.kaggle.com/mlg-ulb/creditcardfraud/tasks?taskId=3444

There are 3 python files:

CreditcardfraudXOX
this is the decision tree model it has an accuracy of 0.9995318516437859

Splitdata
this is the random forest model it has an accuracy of 0.9995669627705019

Cleaneddata
this is the random forest model only with variables that have more 0.75 significance, it has an accuracy of 0.9994733330992591

There are 2 png files:

decisiontree
The decision tree for the first model

featureplot
This is bar plot with the feature importance score for the variables. 




############---CONCLUSION---#############

There is not a big difference between the decision tree and the random forest for this specific data.
The precision obtained is quite similar, but the amount of time to create the model is the relevant key.
The random forest model with clean data has the lowest accuracy (-0.00000936296712428230) compared to
the other 2 models. But it is by far the fastest model. The feature importance graph shows us that 
all of these variables, they do not benefit the accuracy of the model, as they have little importance for the subject.
Therefore, by removing them, the model becomes faster without significantly affecting accuracy.




