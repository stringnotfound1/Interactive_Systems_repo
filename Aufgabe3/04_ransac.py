import numpy as np
import math
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RansacPointGenerator:
    """generates a set points - linear distributed + a set of outliers"""

    def __init__(self, numpointsInlier, numpointsOutlier):
        self.numpointsInlier = numpointsInlier
        self.numpointsOutlier = numpointsOutlier
        self.points = []

        pure_x = np.linspace(0, 1, numpointsInlier)
        pure_y = np.linspace(0, 1, numpointsInlier)
        noise_x = np.random.normal(0, 0.025, numpointsInlier)
        noise_y = np.random.normal(0, 0.025, numpointsInlier)

        outlier_x = np.random.random_sample((numpointsOutlier,))
        outlier_y = np.random.random_sample((numpointsOutlier,))

        points_x = pure_x + noise_x
        points_y = pure_y + noise_y
        points_x = np.append(points_x, outlier_x)
        points_y = np.append(points_y, outlier_y)

        self.points = np.array([points_x, points_y])


class Line:
    """helper class"""

    def __init__(self, a, b):
        # y = mx + b
        self.m = a
        self.b = b


class Ransac:
    """RANSAC class. """

    def __init__(self, points, threshold):
        self.points = points
        self.threshold = threshold
        self.best_model = Line(1, 0)
        self.best_inliers = []
        self.best_score = 1000000000
        self.current_inliers = []
        self.current_model = Line(1, 0)
        self.num_iterations = int(self.estimate_num_iterations(0.99, 0.5, 2))
        self.iteration_counter = 0

    def estimate_num_iterations(self, ransacProbability, outlierRatio, sampleSize):
        """
        Helper function to generate a number of generations that depends on the probability
        to pick a certain set of inliers vs. outliers.
        See https://de.wikipedia.org/wiki/RANSAC-Algorithmus for more information

        :param ransacProbability: std value would be 0.99 [0..1]
        :param outlierRatio: how many outliers are allowed, 0.3-0.5 [0..1]
        :param sampleSize: 2 points for a line
        :return:
        """
        return math.ceil(math.log(1 - ransacProbability) / math.log(1 - math.pow(1 - outlierRatio, sampleSize)))

    def estimate_error(self, p, line):
        """
        Compute the distance of a point p to a line y=mx+b
        :param p: Point
        :param line: Line y=mx+b
        :return:
        """
        return math.fabs(line.m * p[0] - p[1] + line.b) / math.sqrt(1 + line.m * line.m)

    def step(self, iter):
        """
        Run the ith step in the algorithm. Collects self.currentInlier for each step.
        Sets if score < self.bestScore
        self.bestModel = line
        self.bestInliers = self.currentInlier
        self.bestScore = score

        :param iter: i-th number of iteration
        :return:
        """
        self.current_inliers = []
        error_score = 0
        idx = 0

        # sample two random points from point set
        rand_1 = np.random.randint(0, len(self.points[0]) - 1)
        rand_2 = np.random.randint(0, len(self.points[0]) - 1)
        point_1 = self.points[0:, rand_1]
        point_2 = self.points[0:, rand_2]
        # compute line parameters m / b and create new line
        point_1_x, point_1_y = point_1
        point_2_x, point_2_y = point_2
        m = (point_2_y - point_1_y) / (point_2_x - point_1_x)
        b = m * (- point_1_x) + point_1_y
        line = Line(m, b)

        # loop over all points
        # compute error of all points and add to inliers of
        # err smaller than threshold update score, otherwise add error/threshold to score
        error_score = 0
        for idx in range(0, len(self.points[0])):
            point = self.points[0:, idx]
            err = self.estimate_error(point, line)
            if err < self.threshold:
                self.current_inliers.append(point)
            error_score += err

        # if score < self.bestScore: update the best model/inliers/score
        # please do look at resources in the internet :)
        if len(self.current_inliers) > len(self.best_inliers):
            self.best_inliers = self.current_inliers
            self.best_model = line
            self.best_score = error_score

        print(iter, "  :::::::::: bestscore: ", self.best_score, " bestModel: ", self.best_model.m, self.best_model.b)

    def run(self):
        """
        run RANSAC for a number of iterations
        :return:
        """
        for i in range(0, self.num_iterations):
            self.step(i)


rpgs = []
rpg_1 = RansacPointGenerator(100, 45)
print(rpg_1.points)
rpgs.append(rpg_1)

rpg_2 = RansacPointGenerator(60, 120)
print(rpg_2.points)
rpgs.append(rpg_2)

rpg_3 = RansacPointGenerator(120, 60)
print(rpg_3.points)
rpgs.append(rpg_3)

rpg_4 = RansacPointGenerator(50, 50)
print(rpg_4.points)
rpgs.append(rpg_4)

threshold_1 = 0.02
threshold_2 = 0.10
threshold_3 = 0.5
thresholds = [threshold_1, threshold_2, threshold_3]

# print rpg.points.shape[1]
plt.plot(rpg_1.points[0, :], rpg_1.points[1, :], 'ro')

for threshold in thresholds:
    for rpg in rpgs:
        ransac = Ransac(rpg.points, threshold)
        ransac.run()

        plt.figure()
        plt.plot(rpg.points[0, :], rpg.points[1, :], 'ro')
        m = ransac.best_model.m
        b = ransac.best_model.b
        plt.plot([0, 1], [m * 0 + b, m * 1 + b], color='k', linestyle='-', linewidth=2)
        # #
        plt.axis([0, 1, 0, 1])
        title = "Threshold: ", ransac.threshold, "Inliers: ", rpg.numpointsInlier, "Outliers: ", rpg.numpointsOutlier
        plt.title(title)
plt.show()

# Notizen:
# großer Threshold verursacht bei Inliers = Outliers und vor allem Outliers > Inliers Probleme
# kleinere Thresholds haben diese Probleme nicht und unterscheiden sich kaum, Vergleich 0.02 und 0.10
