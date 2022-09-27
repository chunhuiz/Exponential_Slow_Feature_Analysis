import pandas as pd
from sklearn.decomposition import PCA
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import math

class ESFA:
    """
    ESFA algorithm model
    """
    def __init__(self):
        self.EB = None
        self.EZ = None
        self.Diff_Data = None
        self.eig_vector = None
        self.eig_values = None
        self.org_data = None
        self.pca = None

    def whiten_data(self, data):
        # Whitening operation
        self.pca = PCA(whiten=True)
        whiten_data = self.pca.fit_transform(data)
        return whiten_data

    def get_EB(self, data):
        return np.mat(expm(np.dot(data.T, data) / (data.shape[0] - 1)))

    def get_ZB(self, diff_data):
        return np.mat(expm(np.dot(diff_data.T, diff_data) / (diff_data.shape[0] - 1)))

    def get_diff_data(self, data, axis):
        # Difference function
        diff1 = pd.DataFrame(np.diff(data, axis=axis))
        diff2 = pd.DataFrame(data[-1, :] - data[0, :])
        diff = pd.concat([diff2.T, diff1], axis=0)
        return np.mat(diff)

    def fit(self, data_original):
        """
        This function is used for training ESFA model
        :param data_original: training data
        :return: Each slow feature, corresponding eigenvalue and corresponding eigenvector of training data
        """
        self.org_data = data_original
        whiten = self.whiten_data(data_original)
        self.EB = self.get_EB(whiten)
        # if self.EB.T.all() == self.EB.all():
        #     print("yes")
        self.Diff_Data = self.get_diff_data(whiten, axis=0)
        self.EZ = self.get_ZB(self.Diff_Data)
        eig_values, eig_vector = np.linalg.eig(np.dot(self.EB.I, self.EZ))
        sorted_indices = np.argsort(eig_values)
        self.eig_vector = eig_vector[:, sorted_indices]
        self.eig_values = eig_values[sorted_indices]

    def transform(self, data_original):
        """
        This function is used to analyze the test data by using the trained ESFA model
        :param data_original: Test data
        :return: Each slow feature, corresponding eigenvalue and corresponding eigenvector of data
        """
        whiten_data_ = self.pca.transform(data_original)
        feature = np.dot(self.eig_vector.T, np.mat(whiten_data_).T).T
        return feature, self.eig_values, self.eig_vector


def find_kde(input, Confidence):

    """
    KDE nonparametric estimation computes control limit function
    :param1 data_original: one-dimensional nadarray data
    :param2 data_original: The parameter Confidence is the confidence level, which is generally above 0.9
    :return: Specific value of control limit
    """
    plt.figure()
    # ax = sns.kdeplot(input, kernel='Gaussian', bw=((input.max() - input.min()) / 1000))
    ax = sns.kdeplot(input, cumulative=True, kernel='gau'
                     , bw=((input.max() - input.min()) / 1000))
    line = ax.lines[0]
    x1, y1 = line.get_data()
    for i in range(len(y1)):
        if y1[i] > Confidence:
            kzx = x1[i - 1] + (x1[i] - x1[i - 1]) * (y1[i] - Confidence) / (y1[i] - y1[i - 1])
            break
    plt.close()
    return kzx