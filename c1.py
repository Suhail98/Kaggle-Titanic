import pandas as pd
import sklearn
from sklearn import svm
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

data= pd.read_csv("train.csv")


data["Cabin"] = data["Cabin"].fillna('U')
data["Age"] = data["Age"].replace(np.nan, int(data["Age"].mean()))
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
le = preprocessing.LabelEncoder()

print(data.isnull().sum())

data["Sex"] = le.fit(data["Sex"]).transform(data["Sex"])

data["Cabin"] = le.fit(data["Cabin"]).transform(data["Cabin"])
data["Embarked"] = le.fit(data["Embarked"]).transform(data["Embarked"])

data = data.sample(frac=1).reset_index(drop=True)

train, test = train_test_split(data, test_size=0.2)

y = train["Survived"]
x = train[train.columns[~train.columns.isin(['Survived','PassengerId','Name','Ticket'])]]

#clf = LogisticRegression(random_state=0).fit(x, y)


y_test = test["Survived"]
x_test = test[test.columns[~test.columns.isin(['Survived','PassengerId','Name','Ticket'])]]
'''
y_pred = clf.predict(x_test)
print("Number of mislabeled points out of a total %d points : %d"
   % (x_test.shape[0], (y_test != y_pred).sum()))
'''

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(x, y)
print(pipe.score(x_test, y_test)) 


pipe = make_pipeline(StandardScaler(), GaussianNB())
pipe.fit(x, y)
print(pipe.score(x_test, y_test)) 


pipe = make_pipeline(StandardScaler(), svm.SVC())
pipe.fit(x, y)
print(pipe.score(x_test, y_test))

pipe = make_pipeline(StandardScaler(), tree.DecisionTreeClassifier())
pipe.fit(x, y)
print(pipe.score(x_test, y_test))

pipe = make_pipeline(StandardScaler(), neighbors.KNeighborsClassifier(n_neighbors=5))
pipe.fit(x, y)
print(pipe.score(x_test, y_test))