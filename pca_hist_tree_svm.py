import data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import random
from sklearn import svm


# Calculate pca and plot the two dimensional decomposition
def pca_scatter(classes, data):
    scaling = StandardScaler()
    scaling.fit(data)
    scaled_data = scaling.transform(data)
    principal = PCA(n_components=2)
    principal.fit(scaled_data)
    x = principal.transform(scaled_data)
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=classes, cmap='plasma')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.show()


# Plot histogram of attribute values
def display_histogram(data):
    features = list(data.columns)
    for i in features:
        plt.figure()
        if i == "Severity":
            plt.hist(data[i], log=True, bins=[1, 2, 3, 4, 5], rwidth=0.8, align="left")
        else:
            plt.hist(data[i], log=True)
        plt.title([i])
        plt.show()


# Train and return decision tree, predictions, and true values
def train_decision_tree(data):
    training_set = data.copy()
    testing_set = pd.DataFrame(columns=list(data.columns))
    for i in range(int(training_set.index.size / 10000)):
        j = random.randrange(0, training_set.index.size)
        testing_set = testing_set.append(training_set.loc[j])
        testing_set = testing_set.reset_index(drop=True)
        training_set = training_set.drop(j)
        training_set = training_set.reset_index(drop=True)
    samples = training_set[list(training_set.columns[1:])]
    targets = training_set["Severity"]
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(samples, targets)
    prediction = clf.predict(testing_set[list(dataset.columns[1:])])
    return clf, prediction, testing_set["Severity"]


# Train and return a svm model, predictions, and actual values
def train_svm(data):
    training_set = data.copy()
    testing_set = pd.DataFrame(columns=list(data.columns))
    for i in range(int(training_set.index.size / 10000)):
        j = random.randrange(0, training_set.index.size)
        testing_set = testing_set.append(training_set.loc[j])
        testing_set = testing_set.reset_index(drop=True)
        training_set = training_set.drop(j)
        training_set = training_set.reset_index(drop=True)
    samples = training_set[list(training_set.columns[1:])]
    targets = training_set["Severity"]

    svm_model = svm.SVC()
    svm_model.fit(samples.iloc[0:100], targets.iloc[0:100])
    return svm_model, svm_model.predict(testing_set[list(dataset.columns[1:])]), testing_set["Severity"]


dataset = dp.load_data("modified1.csv")
features = list(dataset.columns[1:])
classes = dataset["Severity"]
attributes = dataset[features]
pca_scatter(classes, attributes)
display_histogram(attributes)
display_histogram(pd.DataFrame(classes))
train_decision_tree(dataset)
