import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
import missingno as msno
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

df = pd.read_csv("train.csv")
df.head()

print(df.columns)


print(df.shape)
print(100 * "#")
print(df.info())
print(100 * "#")
print(df.describe())
print(100 * "#")
print(df.head())
print(100 * "#")
print(df.tail())
print(100 * "#")

df.info()

cat_cols = ["target"]
num_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline']
    

def replace_with_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

df.describe().T


plt.figure(figsize=(12,12))
sns.heatmap(df[num_cols].corr(), annot=True)
plt.show()

for col in num_cols:
  sns.distplot(df[col])
  plt.show()
  sns.boxplot(df[col])
  plt.show()

for col in num_cols:
    replace_with_thresholds(df, col, 0.1, 0.9)


for col in cat_cols:
  sns.countplot(x=col, data=df)
  plt.show()

df["target"].value_counts()


df.isnull().sum()

msno.bar(df)

msno.heatmap(df)


df.head()


x = df.drop("target",axis =1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = DecisionTreeClassifier()


model.fit(x_train, y_train)
preds = model.predict(x_test)
f1_score(y_true=y_test, y_pred=preds, average='weighted')
precision_score(y_true=y_test, y_pred=preds, average='weighted')
recall_score(y_true=y_test, y_pred=preds, average='weighted')
accuracy_score(y_true=y_test, y_pred=preds)


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

best_model = GridSearchCV(model, params, cv=10, n_jobs=-1, verbose=False)
best_model.fit(x_train, y_train)

final_model = best_model.best_estimator_
final_model.fit(x_train, y_train)
cv_results = cross_validate(final_model, x_train, y_train, cv=10, scoring=["accuracy"])


final_preds = final_model.predict(x_test)
f1_score(y_test, final_preds, average='weighted')
precision_score(y_test, final_preds, average='weighted')
recall_score(y_test, final_preds, average='weighted')
accuracy_score(y_test, final_preds)

plt.figure(figsize=(15,15))
tree.plot_tree(final_model)
plt.show()
print(classification_report(y_test,final_preds))


feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': x_train.columns})
plt.figure(figsize=(10, 10))
sns.set(font_scale=1)
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(
    by="Value", ascending=False)[0:10])
plt.title('Features')
plt.tight_layout()
plt.show()

print(final_model.get_params())
