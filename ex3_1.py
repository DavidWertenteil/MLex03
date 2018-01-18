__author__ = 'davidwer'
__author2__ = 'omersc'
# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from os import path
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
import time
import pickle
# #############################################################################
# Setting up


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


# Load Data
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')

X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# ########################### 1 #############################################
time_each_training = []
precisions_RBM = []
precisions_raw = []


def training(n, classify):
    filename = path.join("trained/", "{0}_components.sav".format(n**2))
    # Training
    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = n ** 2
    logistic.C = 6000.0
    # Training RBM-Logistic Pipeline
    start_time = time.clock()
    classify.fit(X_train, Y_train)
    time_each_training.append(time.clock() - start_time)
    # Save features to file for later plotting
    pickle.dump(classify, open(filename, 'wb'))

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)
    precisions_RBM.append(precision_score(Y_test, classify.predict(X_test),
                                          average='macro'))
    #  ----------------------------- Plotting -------------------------------
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(n, n, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(str(n**2) + ' components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.savefig(path.join("plots/", "{0}_components.png".format(n**2)))


ran = range(2, 21)
# for i in ran:
#     training(i, classifier)

# Save data to files
# pickle.dump(time_each_training, open("trained/time_array.sav", 'wb'))
# pickle.dump(precisions_RBM, open("trained/precisions_RBM_array.sav", 'wb'))

# For testing - Load features to plot:
time_each_training = pickle.load(open("trained/time_array.sav", 'rb'))
precisions_RBM = pickle.load(open("trained/precisions_RBM_array.sav", 'rb'))

plt.plot(time_each_training, precisions_RBM, 'b',
         [0, time_each_training[-1]],
         [0.77, 0.77],
         'r')
plt.title("RBM vs Time")
plt.xlabel("Time")
plt.ylabel("Precisions RBM")
plt.savefig(path.join("plots/", "RBM vs Time.png"))
plt.close()

ran = [i**2 for i in ran]
plt.plot(time_each_training, ran, 'b')
plt.title("Precisions vs Time")
plt.xlabel("Time")
plt.ylabel("Precision Score")
plt.savefig(path.join("plots/", "Precision Score vs Time"))

