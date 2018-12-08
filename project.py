# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:23:42 2018

@author: dujun
"""

import pandas as pd
import numpy as np
import random as rnd
import re as re

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

#导入数据
train =pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#整合数据
full_data = [train, test]

#浏览数据
print(train.columns.values)
train.head()
train.info()
test.info()
train.describe()



###Train set analyze
###one factor analyze
###1.total survival
fig,ax = plt.subplots(figsize=(9,7))
train["Survived"].value_counts().plot(kind="bar")
ax.set_xticklabels(("Not Survived","Survived"),  rotation= "horizontal" )
ax.set_title("Bar plot of Survived ")
###可以发现，大部分乘客没有获救

###2.age and fare
fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(10,15))
train["Age"].hist(ax=ax[0])
ax[0].set_title("Hist plot of Age")
train["Fare"].hist(ax=ax[1])
ax[1].set_title("Hist plot of Fare")
###可以发现，大部分乘客的年龄位于20-40岁之间，总体上呈正态分布。大部分乘客的票价很低，位于0-100之间，其他少部分乘客的票价较高。

###3.sex
fig,ax = plt.subplots(figsize=(9,7))
train["Sex"].value_counts().plot(kind="bar")
ax.set_xticklabels(("male","female"),rotation= "horizontal"  )
ax.set_title("Bar plot of Sex ")
###大多乘客为男性

###4.pclass
fig,ax = plt.subplots(figsize=(9,7))
train["Pclass"].value_counts().plot(kind="bar")
ax.set_xticklabels(("Class1","Class2","Class3"),rotation= "horizontal"  )
ax.set_title("Bar plot of Pclass ")
###大部分乘客位于第三等级，第一等级和第二等级的乘客各有200个左右。




###5.SibSb
fig,ax = plt.subplots(figsize=(9,5))
train["SibSp"].value_counts().plot(kind="bar")
ax.set_title("Bar plot of SibSp ")
###大部分乘客在船上没有兄弟姐妹或配偶，大约200位乘客在船上有1个兄弟姐妹或配偶

###6.Parch
fig,ax = plt.subplots(figsize=(9,5))
train["Parch"].value_counts().plot(kind="bar")
ax.set_title("Bar plot of Parch ")
###大部分乘客在船上没有父母或子女，100多位乘客在船上有1个兄弟姐妹或配偶，大约90位乘客在船上有2个兄弟姐妹或配偶

###7.出发港口 embarked
fig,ax = plt.subplots(figsize=(9,5))
train["Embarked"].value_counts().plot(kind="bar")
ax.set_xticklabels(("Southampton","Cherbourg","Queenstown"),rotation= "horizontal"  )
ax.set_title("Bar plot of Embarked ")
###可以发现，大部分乘客从Southampton港口出发，不到200位乘客从Cherburge出发，不到100位乘客从Queentown出发


###Analyze by pivoting features

###1.Pclass(no missing value) impact on train set
pd.crosstab(train["Pclass"],train["Survived"])
pd.crosstab(train["Pclass"],train["Survived"]).plot(kind="bar")
pclass_rate = train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
pclass_rate
sns.factorplot('Pclass','Survived',order=[1,2,3], color='lawngreen',data=pclass_rate,size=8)
#Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
#Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
#Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
#Pclass varies in terms of Age distribution of passengers.
#Decisions.
#Consider Pclass for model training.

###2.Sex
pd.crosstab(train["Sex"],train["Survived"])
pd.crosstab(train["Sex"],train["Survived"]).plot(kind="bar", color=['limegreen','tomato'],figsize=(10,7))
sex_rate = train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sex_rate
sex_rate_bar = sex_rate['Survived'].plot.bar(figsize=(11,8),tick_label=(sex_rate['Sex']),color=['limegreen','darkorange'])
sex_rate_bar_label=sex_rate_bar.get_xticks().tolist()
sex_rate_bar_label[0]="Female"
sex_rate_bar_label[1]="Male"
sex_rate_bar.set_xticklabels(sex_rate_bar_label)
sex_rate_bar.set_title('Sex VS. Survival Rate')
sex_rate_bar.set_ylabel('Survival Rate')
sex_rate_bar.set_xlabel('Sex')
#Female passengers had much better survival rate than males. Confirms classifying
#Add Sex feature to model training.


###3.SibSp and Parch
###3.1.Sibsp
pd.crosstab(train["SibSp"],train["Survived"])
pd.crosstab(train["SibSp"],train["Survived"]).plot(kind="bar",color=['limegreen','tomato'],figsize=(10,7))
sibsp_rate = train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sibsp_rate

###3.2.Parch
pd.crosstab(train["Parch"],train["Survived"])
pd.crosstab(train["Parch"],train["Survived"]).plot(kind="bar",color=['limegreen','tomato'],figsize=(10,7))
parch_rate = train[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
parch_rate
### With the number of siblings/spouse and the number of children/parents we can create new feature called Family Size
for dataset in full_data:
    dataset['FamilySize']= dataset['SibSp']+ dataset['Parch']+1
print(train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived', ascending=False))
pd.crosstab(train["FamilySize"],train['Survived']).plot(kind="bar",color=['limegreen','tomato'],figsize=(10,7))

### it seems has a good effect on our prediction but let's go further and categorize people to check whether they are alone in this ship or not
### creat new feature IsAlone
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1
isalone_rate = train[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(train["IsAlone"],train["Survived"]).plot(kind="bar",color=['limegreen','tomato'],figsize=(10,7))
isalone_rate_bar = isalone_rate['Survived'].plot.bar(fc='forestgreen',figsize=(11,8))
isalone_rate_bar.set_title('IsAlone VS. Survival Rate')
isalone_rate_bar.set_ylabel('Survival Rate')
isalone_rate_bar.set_xlabel('IsAlone')
###good! the impact is considerable

###4.Embarked
### the embarked feature has some missing value. and we try to fill those with the most occurred value ( 'S' )
for dataset in full_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
embarked_rate = train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived', ascending=False)
embarked_rate
###4.1. Embarked,Sex and Pclass VS. Survival Rate
grid = sns.FacetGrid(train, row='Embarked', size=4.4, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', hue_order=['female','male'])
grid.add_legend()
###Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
###Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
###Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
###decision: Complete and add Embarked feature to model training.


###5.Fare
### Fare also has some missing value and we will replace it with the median. then we categorize it into 4 ranges
for dataset in full_data:
    dataset['Fare']=dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare']=pd.qcut(train['Fare'],4)
fare_rate = train[['CategoricalFare','Survived']].groupby(['CategoricalFare']).mean()
fare_rate
fare_rate_bar = fare_rate['Survived'].plot.bar(color=['orangered','tomato','lightsalmon','mistyrose'],figsize=(10,7))
fare_rate_bar_label=fare_rate_bar.get_xticks().tolist()
fare_rate_bar_label[0]="(0, 7.91]"
fare_rate_bar_label[1]="(7.91, 14.454]"
fare_rate_bar_label[2]="(14.454, 31.0]"
fare_rate_bar_label[3]="(31.0, 512.329]"
fare_rate_bar.set_xticklabels(fare_rate_bar_label)
fare_rate_bar.set_title('Fare VS. Survival Rate')
fare_rate_bar.set_ylabel('Survival Rate')
fare_rate_bar.set_xlabel('CategoricalFare')


###6.Age
### we have plenty of missing values in this feature. # generate random numbers between (mean - std) and (mean + std). then we categorize age into 5 range
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg-age_std, age_avg+age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)
train["CategoricalAge"].value_counts().plot(kind="bar",figsize=(10,8))
age_rate = train[['CategoricalAge','Survived']].groupby(['CategoricalAge']).mean()
age_rate
#barchart
age_rate_bar = age_rate['Survived'].plot.bar(color=['dodgerblue','deepskyblue','lightskyblue','skyblue','lightblue'],figsize=(11,8))
age_rate_bar_label=age_rate_bar.get_xticks().tolist()
age_rate_bar_label[0]="(0,16]"
age_rate_bar_label[1]="(16,32]"
age_rate_bar_label[2]="(32,48]"
age_rate_bar_label[3]="(48,64]"
age_rate_bar_label[4]="(64,80]"
age_rate_bar.set_xticklabels(age_rate_bar_label)
age_rate_bar.set_title('Age VS. Survival Rate')
age_rate_bar.set_ylabel('Survival Rate')
age_rate_bar.set_xlabel('CategoricalAge')

#piechart
labels=["(0,16]","(48,64]","(32,48]","(16,32]","(64,80]"]
age_rate_pie = age_rate['Survived'].plot.pie(subplots=True,autopct='%.2f%%',colors=['red','yellowgreen','dodgerblue','lime','gold'], labels=labels,figsize=(9, 9))[0]
age_rate_pie.set_title('Proportion Of Survival Rate in Age Group')
age_rate_pie.legend(loc=('lowerleft'))


###7.Name
### in this feature we can find the title of people
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.',name)
    # if the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return""

for dataset in full_data:
    dataset['Title']=dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'],train['Sex']))
### now we have titles. let's categorize it and check the title impact on survival rate
for dataset in full_data:
    dataset['Title']=dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Rev','Sir','Dona'],'Rare')
    dataset['Title']=dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')

name_rate = train[['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived',ascending=False)
name_rate
labels=["Mrs","Miss","Master","Rare","Mr"]
name_rate_pie = name_rate['Survived'].plot.pie(subplots=True,autopct='%.2f%%', colors=['orange','lime','red','yellow','royalblue'],labels=labels,figsize=(9, 9))[0]
name_rate_pie.set_title('Proportion Of Survival Rate in Name Group')
name_rate_pie.legend(loc=('lowerleft'))



### Data cleaning
### Clean data and map features into numerical values
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

print (train.head(10))
#train = train.values
#test  = test.values

### Model, predict and solve


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.copy()
X_train.shape, Y_train.shape, X_test.shape

###logistic regression CV

#Logistic Regression is a useful model to run early in the workflow.
#Logistic regression measures the relationship between the categorical dependent
#variable (feature) and one or more independent variables (features) by estimating probabilities
#using a logistic function, which is the cumulative logistic distribution. Reference Wikipedia.


logreg = LogisticRegression()
tuned_parameters=[{'penalty':['l1','l2'],
                   'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                    'solver':['liblinear'],
                    'multi_class':['ovr']},
                {'penalty':['l2'],
                 'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
                'solver':['lbfgs'],
                'multi_class':['ovr','multinomial']}]
model1= GridSearchCV(LogisticRegression(tol=1e-6),tuned_parameters,cv=5)
model1.fit(X_train, Y_train)
print('Grid best parameter (max. accuracy): ', model1.best_params_)
print('Grid best score (accuracy): ', model1.best_score_)


# Support Vector Machines
svc = SVC()
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
model2= GridSearchCV(SVC(),tuned_parameters,cv=5)
model2.fit(X_train, Y_train)
print('Grid best parameter (max. accuracy): ', model2.best_params_)
print('Grid best score (accuracy): ', model2.best_score_)


# k-Nearest NeighborsCV
knn = KNeighborsClassifier()
grid_values = {'n_neighbors': [2,3,4,5,6,7,8,9,10]}
# default metric to optimize over grid parameters: accuracy
model3 = GridSearchCV(knn, param_grid = grid_values)
model3.fit(X_train, Y_train.reshape(-1))
print('Grid best parameter (max. accuracy): ', model3.best_params_)
print('Grid best score (accuracy): ', model3.best_score_)



# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
model4_gaussian = round(gaussian.score(X_train, Y_train), 2)
model4_gaussian


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
model5_sgd = round(sgd.score(X_train, Y_train), 2)
model5_sgd


# Decision Tree
decision_tree = DecisionTreeClassifier()
grid_values = [
{'max_features':[2, 4, 6, 7], 'max_depth': [10,20,30,40,50,60,70,80,90,100] }]
# default metric to optimize over grid parameters: accuracy
model6 = GridSearchCV(decision_tree, param_grid = grid_values,cv=5)
model6.fit(X_train, Y_train)
print('Grid best parameter (max. accuracy): ', model6.best_params_)
print('Grid best score (accuracy): ', model6.best_score_)

#Random Forest
random_forest = RandomForestClassifier()
parameters = {
    "n_estimators": [10, 15, 20],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6],
}
model7 = GridSearchCV(random_forest, parameters, cv=5)
model7.fit(X_train, Y_train)
print('Grid best parameter (max. accuracy): ', model7.best_params_)
print('Grid best score (accuracy): ', model7.best_score_)


#model evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [model1.best_score_,model2.best_score_, model3.best_score_,
              model4_gaussian, model5_sgd,
               model6.best_score_, model7.best_score_]})
models.sort_values(by='Score', ascending=False)



###predict
tree=DecisionTreeClassifier(max_depth=90, max_features=4)
tree.fit(X_train,Y_train)
Y_pred = tree.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('predict_survive.csv',index=False)
