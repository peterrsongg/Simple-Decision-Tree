#4a
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

Data = load_breast_cancer()

X = Data.data
Y = Data.target
#plot accuracy of a scikit-learn Decision Tree Classifier as a function of the max _leaf_nodes parameter 
# use the breast cancer data set that is provide ,and the specific max_leaf_nodes values 2^n for n: 1 to 5.
leafNodes = [1,2,3,4,5]


# For each of these numbers of leaf nodes, compute two distinct accuracy values: first, the training accuracy, 
fullDataScores = []
crossValAccuracy = []
for i in leafNodes:
    #train the model without cross validation
    clf = DecisionTreeClassifier(max_leaf_nodes = 2**i)
    clf.fit(X,Y)
    fullDataScores.append(clf.score(X,Y))
    total = 0
    # train the model with cross validation
    clf = DecisionTreeClassifier(max_leaf_nodes = 2**i)
    #iterate 4 folds 10 times
    for j in range(10):
        #take the mean of the 4 folds 
        score = np.mean(cross_val_score(clf,X,Y,cv = 4))
        total += score
    # take the mean of the scores
    crossValAccuracy.append(total/10)
    
    
# report and plot both accuracy values as functions of max_leaf_nodes. 
X2 = [2**i for i in leafNodes]
Y2 = fullDataScores
Z = crossValAccuracy

fig,axs = plt.subplots(2,figsize=(15,15))
fig.suptitle("Accuracy vs max_leaf_nodes")
fig.subplots_adjust(hspace=.5)
axs[0].scatter(X2,Y2)
axs[0].set_title('No Cross Validation')
axs[0].set_xlabel("# of Leaf Nodes")
axs[0].set_ylabel("Accuracy")
axs[1].scatter(X2,Z)
axs[1].set_title("Cross Validation")
axs[1].set_xlabel("# of Leaf Nodes")
axs[1].set_ylabel("Accuracy")


