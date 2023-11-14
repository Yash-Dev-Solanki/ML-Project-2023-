import numpy as np
import numpy.typing as npt
import pandas as pd
import os

from sklearn.svm import SVC
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics import accuracy_score


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

    def run(self, num_gen: int):
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

            self.best = np.max(self.fitness_arr)
            print(f"{gen}th generation best gene accuracy is: {self.best * 100} %")

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
                    ptr += 1

            for idx, fit in enumerate(child_fitness):
                if fit > threshold:
                    next_gen[ptr] = child_fitness[idx]
                    ptr += 1

            self.current_gen = next_gen



    @staticmethod
    def fitness(gene: npt.ArrayLike) -> float:
        c = gene[0]
        gamma = gene[1]
        svc = SVC(kernel='rbf', C=c, gamma=gamma)
        svc.fit(x_train, y_train)
        predicted = svc.predict(x_test)

        return accuracy_score(y_test, predicted)

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

print(y_test)

GA = GeneticAlgorithm()
GA.initialize(8)
GA.run(5)
print(GA.best)