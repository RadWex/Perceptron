import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class Point():
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_label(self):
        return self.label


class Perceptron():
    def __init__(self):
        self.W_x = random.uniform(0, 1.0)
        self.W_y = random.uniform(0, 1.0)
        self.W_b = random.uniform(0, 1.0)

    def train(self, collection, adaptive=1):
        solutions = 0
        for point in collection:
            a = self.W_x * point.x + self.W_y * \
                point.y + self.W_b

            #e_i = t_i - y_i
            error = point.get_label() - self.activation(a)
            if error == 0.0:
                solutions += 1
            else:
                self.W_x += error * point.x * adaptive
                self.W_y += error * point.y * adaptive
                self.W_b = self.W_b + error

        return solutions/len(collection)*100

    def test(self, collection, adaptive=0):
        solutions = 0
        for point in collection:
            a = self.W_x * point.x + self.W_y * \
                point.y + self.W_b
            error = point.get_label() - self.activation(a)
            if error == 0.0:
                solutions += 1

        return solutions/len(collection)*100

    def activation(self, a):
        if(a >= 0):
            return 1
        else:
            return 0


def randPoint(range):
    x = random.randint(range[0], range[1])
    y = random.randint(range[0], range[1])
    label = 0
    if(x <= y):
        label = 1
    return Point(x, y, label)


def learn(iterations, amount_of_points, debug=False):
    points_range = (-50, 50)
    points = []
    training = []
    testing = []
    for i in range(0, amount_of_points):
        points.append(randPoint(points_range))

    cut = int(0.8 * len(points))  # 80% of the list
    random.shuffle(points)
    training = points[:cut]  # first 80% of shuffled list
    testing = points[cut:]  # last 20% of shuffled list

    percept = Perceptron()
    trained_iter = None
    for i in range(1, iterations+1):
        train_output = percept.train(training, adaptive=0.25*(1/i))
        #train_output = percept.train(training)
        test_output = percept.test(testing)
        if debug:
            print("Epoch {:2d} Efficiency training {:3.2f}'%' Efficiency tests {:.2f}%".format(
                i, train_output, test_output))
        if not debug and train_output == 100:
            trained_iter = i
            break
    if trained_iter is None:
        trained_iter = iterations+1
    return trained_iter


def drawPlot():
    tmp = []
    for i in range(1, 6):
        tmp.append(learn(25, pow(10, i)))
    # chart

    x = np.arange(5)
    labels = (10, 100, 1000, 10000, 100000)
    fig, ax = plt.subplots()
    plt.title(
        "The impact of the number of points on learning")
    # (iteration=iteration in which 100 '%' efficiency was achieved)
    plt.bar(x, tmp)
    plt.xticks(x, labels)

    plt.xlabel('Points')
    plt.ylabel('Iteration')
    # the perceptron is the worst at 100 points, more points give
    # faster results and is more stable, with fewer points it heavily depends
    # on what points will be drawn
    plt.show()


if __name__ == "__main__":
    # function x = y
    learn(25, 100, debug=True)
    drawPlot()
