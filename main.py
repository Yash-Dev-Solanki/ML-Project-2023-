import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
import os

from sklearn.svm import SVC
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils._param_validation import InvalidParameterError

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class GeneticAlgorithm:
    # implements a genetic algorithm to optimize the
    # hyperparameters C & gamma for SVM

    def __init__(self):
        self.fitness_arr = None
        self.best = 0
        self.pop_size = 0
        self.current_gen = None

        self.c_choice = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.gamma_choice = [0.000030517578125, 0.00006103515625, 0.0001220703125, 0.0001220703125, 0.00048828125,
                             0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25,
                             0.5, 1, 2, 4, 8]

    def initialize(self, pop_size: int):
        # initialize a starting population of pop_size
        self.current_gen = np.zeros((pop_size, 2))
        self.pop_size = pop_size

        for i in range(self.pop_size):
            c = np.random.choice(self.c_choice)
            gamma = np.random.choice(self.gamma_choice)
            self.current_gen[i] = np.array([c, gamma]).reshape(1, 2)

        self.fitness_arr = np.array([self.fitness(gene) for gene in self.current_gen])
        self.best = np.max(self.fitness_arr)
        print(f"Initial best accuracy : {self.best * 100} %")

    def run(self, num_gen: int) -> npt.ArrayLike:
        for gen in range(num_gen):
            # create next generation by selection
            children = np.zeros(np.shape(self.current_gen))
            for i in range(0, self.pop_size, 2):
                parent1 = self.selection(self.current_gen, self.fitness_arr)
                parent2 = self.selection(self.current_gen, self.fitness_arr)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                children[i] = child1
                children[i + 1] = child2

            # implemented elitism replacement
            # take best in children & parents
            child_fitness = np.array([self.fitness(gene) for gene in children])
            sorted_fit = np.sort(np.append(self.fitness_arr, child_fitness))

            threshold = sorted_fit[self.pop_size - 1]
            next_gen = np.zeros((self.pop_size, 2))

            ptr = 0
            for idx, fit in enumerate(self.fitness_arr):
                if fit > threshold:
                    next_gen[ptr] = self.current_gen[idx]
                    self.fitness_arr[ptr] = fit
                    ptr += 1

            for idx, fit in enumerate(child_fitness):
                if fit > threshold:
                    next_gen[ptr] = child_fitness[idx]
                    self.fitness_arr[ptr] = fit
                    ptr += 1

            self.current_gen = next_gen
            self.best = np.max(self.fitness_arr)
            print(f"{gen + 1}th generation best gene accuracy is: {self.best * 100} %")

        return self.current_gen[np.argmax(self.fitness_arr)]

    @staticmethod
    def fitness(gene: npt.ArrayLike) -> float:
        c = gene[0]
        gamma = gene[1]
        try:
            svc = SVC(kernel='rbf', C=c, gamma=gamma)
            svc.fit(x_train, y_train)
            predicted = svc.predict(x_test)
            return accuracy_score(y_test, predicted)
        except InvalidParameterError:
            return 0

    def selection(self, parents: npt.ArrayLike, fitness_arr: npt.ArrayLike) -> npt.ArrayLike:
        # implement tournament selection with s = 2
        idx1 = np.random.choice(self.pop_size)
        idx2 = np.random.choice(self.pop_size)

        if fitness_arr[idx1] > fitness_arr[idx2]:
            return parents[idx1]
        else:
            return parents[idx2]

    @staticmethod
    def crossover(parent1: npt.ArrayLike, parent2: npt.ArrayLike) -> npt.ArrayLike:
        mixing_prob = 0.5
        prob1 = np.random.random()
        prob2 = np.random.random()

        children = np.zeros((2, 2))

        if prob1 < mixing_prob:
            children[0, 0] = parent1[0]
            children[1, 0] = parent2[0]
        else:
            children[0, 0] = parent2[0]
            children[1, 0] = parent1[0]

        if prob2 < mixing_prob:
            children[0, 1] = parent1[1]
            children[1, 1] = parent2[1]
        else:
            children[0, 1] = parent2[1]
            children[1, 1] = parent1[1]

        return children

    def mutation(self, child: npt.ArrayLike) -> npt.ArrayLike:
        p1 = np.random.random()
        p2 = np.random.random()
        c = child[0]
        gamma = child[1]

        if p1 < 0.2:
            c = np.random.choice(self.c_choice)

        if p2 < 0.2:
            gamma = np.random.choice(self.gamma_choice)

        return np.array([c, gamma])


class DifferentialEvolution:

    def __init__(self):
        self.fitness_arr = None
        self.current_gen = None
        self.pop_size = 0
        self.c_choice = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.gamma_choice = [0.000030517578125, 0.00006103515625, 0.0001220703125, 0.0001220703125, 0.00048828125,
                             0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25,
                             0.5, 1, 2, 4, 8]
        self.max_fit = 0
        self.cross_prob = 0.5

    def initialize(self, pop_size):
        self.pop_size = pop_size
        self.current_gen = np.zeros((pop_size, 2))

        for i in range(self.pop_size):
            c = np.random.choice(self.c_choice)
            gamma = np.random.choice(self.gamma_choice)
            self.current_gen[i] = np.array([c, gamma]).reshape(1, 2)

        self.fitness_arr = np.array([self.fitness(gene) for gene in self.current_gen])
        self.max_fit = np.max(self.fitness_arr)
        print(f"Initial accuracy {self.max_fit * 100}")

    def run(self, num_gen):
        for gen in range(num_gen):
            for idx, gene in enumerate(self.current_gen):
                mutated = self.mutation(idx, self.fitness_arr[idx])
                new_gene = self.crossover(gene, mutated)

                if (new_gene == gene).all():
                    continue

                new_fit = self.fitness(new_gene)

                if new_fit > self.fitness_arr[idx]:
                    self.current_gen[idx] = new_gene
                    self.fitness_arr[idx] = new_fit
                    self.max_fit = new_fit

            print(f"{gen + 1} accuracy is {self.max_fit * 100}")

        return self.current_gen[np.argmax(self.fitness_arr)]

    @staticmethod
    def fitness(gene: npt.ArrayLike) -> float:
        c = gene[0]
        gamma = gene[1]
        try:
            svc = SVC(kernel='rbf', C=c, gamma=gamma)
            svc.fit(x_train, y_train)
            predicted = svc.predict(x_test)
            return accuracy_score(y_test, predicted)
        except InvalidParameterError:
            return 0

    def mutation(self, idx: int, fitness: float) -> npt.ArrayLike:
        fp = (0.9 * (fitness / self.max_fit)) + 0.1
        F = (2.2 - fp) * np.random.uniform(low=-0.5, high=0.5)

        r1 = np.random.choice([i for i in range(self.pop_size) if i not in [idx]])
        r2 = np.random.choice([i for i in range(self.pop_size) if i not in [r1, idx]])
        r3 = np.random.choice([i for i in range(self.pop_size) if i not in [r1, r2, idx]])

        return self.current_gen[r1] + (F * (self.current_gen[r2] - self.current_gen[r3]))

    def crossover(self, gene: npt.ArrayLike, mutated: npt.ArrayLike) -> npt.ArrayLike:
        rand = np.random.random()

        if rand <= self.cross_prob:
            return mutated

        else:
            return gene


categories = ["glioma", "meningioma", "notumor", "pituitary"]

# creating training dataframe
flat_data_arr = []  # input array
target_arr = []  # output array
train_dir = r"archive\Training"

for i in categories:
    path = os.path.join(train_dir, i)

    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (10, 10, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(categories.index(i))

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
training_data = pd.DataFrame(flat_data)
training_data['Target'] = target
x_train = training_data.iloc[:, :-1]
y_train = training_data.iloc[:, -1]

# creating testing dataframe
flat_data_arr = []  # input array
target_arr = []  # output array
test_dir = r"archive\Testing"

for i in categories:
    path = os.path.join(test_dir, i)

    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (10, 10, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(categories.index(i))

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
testing_data = pd.DataFrame(flat_data)
testing_data['Target'] = target
x_test = testing_data.iloc[:, :-1]
y_test = testing_data.iloc[:, -1]

# define cross-validation method to use
cv = KFold(n_splits=5, random_state=1, shuffle=True)

np.set_printoptions(precision=2)

t_start = datetime.datetime.now()
DE = DifferentialEvolution()
DE.initialize(16)
best_params = DE.run(20)
t_end = datetime.datetime.now()
print(f"Training time: {t_end - t_start}")
print(best_params)

# build regression model
model = SVC(C=best_params[0], gamma=best_params[1], kernel='rbf', probability=True)
model.fit(x_train, y_train)

# accuracy on training data
predict = model.predict(x_train)
print("%0.2f accuracy on training" % (accuracy_score(y_train, predict)))

# use k-fold CV to evaluate model
scores = cross_val_score(model, x_train, y_train, cv=cv, n_jobs=-1)

# get recall & precision
predicted = model.predict(x_test)
print(f"{recall_score(y_test, predicted, average='macro')} recall score")
print(f"{precision_score(y_test, predicted, average='macro')} precision score")

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# create confusion matrix

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels= categories,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()



