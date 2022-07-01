# ライブラリをインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn import svm, metrics

# irisデータ(csv形式)から読み込む
iris_df = pd.read_csv('iris.csv')
iris_df = iris_df[['sepal length (cm)', 'petal length (cm)', 'species']]

# setosa x versicolor のデータフレームを作成
iris_df_1 = iris_df[iris_df['species'] != 'Iris-virginica']
iris_data_1 = iris_df_1[['sepal length (cm)', 'petal length (cm)']]
iris_target_1 = iris_df_1['species']

# versicolor x virginica のデータフレームを作成
iris_df_2 = iris_df[iris_df['species'] != 'Iris-setosa']
iris_data_2 = iris_df_2[['sepal length (cm)', 'petal length (cm)']]
iris_target_2 = iris_df_2['species']

# versicolor x virginica について、SVMを作成して識別精度を確認
clf_1 = svm.SVC(kernel="poly", degree=3, C=1)
clf_1 = clf_1.fit(iris_data_2, iris_target_2)
predict = clf_1.predict(iris_data_2)
print('poly SVM C=1 : accuracy_score', metrics.accuracy_score(iris_target_2, predict))
print('poly SVM C=1 : f1_score', metrics.f1_score(iris_target_2, predict, average="micro"))

clf_2 = svm.SVC(kernel="poly", degree=3, C=10)
clf_2 = clf_2.fit(iris_data_2, iris_target_2)
predict = clf_2.predict(iris_data_2)
print('poly SVM C=10 : accuracy_score', metrics.accuracy_score(iris_target_2, predict))
print('poly SVM C=10 : f1_score', metrics.f1_score(iris_target_2, predict, average="micro"))

clf_3 = svm.SVC(kernel="rbf", gamma=1, C=1)
clf_3 = clf_3.fit(iris_data_2, iris_target_2)
predict = clf_3.predict(iris_data_2)
print('RBF SVM gamma=1 : accuracy_score', metrics.accuracy_score(iris_target_2, predict))
print('RBF SVM gamma=1 : f1_score', metrics.f1_score(iris_target_2, predict, average="micro"))

clf_4 = svm.SVC(kernel="rbf", gamma=10, C=1)
clf_4 = clf_4.fit(iris_data_2, iris_target_2)
predict = clf_4.predict(iris_data_2)
print('RBF SVM gamma=10 : accuracy_score', metrics.accuracy_score(iris_target_2, predict))
print('RBF SVM gamma=10 : f1_score', metrics.f1_score(iris_target_2, predict, average="micro"))

# 散布図表示のコード
x_min, x_max = iris_data_2['sepal length (cm)'].min() - 0.5, iris_data_2['sepal length (cm)'].max() + 0.5
y_min, y_max = iris_data_2['petal length (cm)'].min() - 0.5, iris_data_2['petal length (cm)'].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
mesh_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['sepal length (cm)', 'petal length (cm)'])

# figure = plt.figure(figsize=(15, 5))
# ax = plt.subplot(1, 5, 1)
# ax.scatter(iris_df_2[iris_df_2['species']=='Iris-versicolor']['sepal length (cm)'], 
#            iris_df_2[iris_df_2['species']=='Iris-versicolor']['petal length (cm)'], c='red')
# ax.scatter(iris_df_2[iris_df_2['species']=='Iris-virginica']['sepal length (cm)'], 
#            iris_df_2[iris_df_2['species']=='Iris-virginica']['petal length (cm)'], c='blue')
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())

ax = plt.subplot(1, 2, 1)
Z = clf_1.decision_function(mesh_df)
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-versicolor']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-versicolor']['petal length (cm)'], c='red')
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-virginica']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-virginica']['petal length (cm)'], c='blue')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax = plt.subplot(1, 2, 2)
Z = clf_2.decision_function(mesh_df)
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-versicolor']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-versicolor']['petal length (cm)'], c='red')
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-virginica']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-virginica']['petal length (cm)'], c='blue')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

plt.show()

ax = plt.subplot(1, 2, 1)
Z = clf_3.decision_function(mesh_df)
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-versicolor']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-versicolor']['petal length (cm)'], c='red')
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-virginica']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-virginica']['petal length (cm)'], c='blue')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax = plt.subplot(1, 2, 2)
Z = clf_4.decision_function(mesh_df)
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-versicolor']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-versicolor']['petal length (cm)'], c='red')
ax.scatter(iris_df_2[iris_df_2['species']=='Iris-virginica']['sepal length (cm)'], 
           iris_df_2[iris_df_2['species']=='Iris-virginica']['petal length (cm)'], c='blue')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

plt.show()