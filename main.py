import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title('Streamlit Exemple')
st.write('Explore different classifiers')

dataset_name = st.sidebar.selectbox(
    'Select a dataset',
    ('Iris', 'Breast Cancer', 'Wine Dataset')
)

classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

def get_dataset(name):
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y

x, y = get_dataset(dataset_name)

st.write('Shape of dataset:', x.shape)
st.write('Number of classes:', len(np.unique(y)))

def add_parameter(clf_name):
    params = dict()
    if clf_name == 'KNN':
        k = st.sidebar.slider('K (number of neighbors)', 1, 15)
        params['K'] = k
    elif clf_name == 'SVM':
        c = st.sidebar.slider('C (Regularization parameter)', 0.01, 10.0, step=0.01)
        params['C'] = c
    else:  
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params = add_parameter(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=1234
        )
    return clf

clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.write(f'You selected: **{dataset_name}** and **{classifier_name}**')
st.write('Parameters:', params)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc:.2f}')

pca = PCA(n_components=2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(plt)
