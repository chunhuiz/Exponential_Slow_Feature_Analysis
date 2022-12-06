import matplotlib.pyplot as plt
from utils import *
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['axes.unicode_minus']=False

if __name__ == "__main__":
    """ 
    .npy file is the input data
    feature_train is the training data under normal condition
    feature_test_normal is the test data under normal condition
    feature_test_fault is the test data of the fault conditions
    """
    data_train = np.load('./data/data_train.npy')
    data_test_normal = np.load('./data/data_test_normal.npy')
    data_test_fault = np.load('./data/data_test_fault.npy')

    # Model Initialization
    model = ESFA()

    # Model training
    model.fit(data_original=data_train)

    # test data Transformation
    feature_normal_train, _, _ = model.transform(data_train)
    feature_normal, values_normal, vector_normal = model.transform(data_test_normal)
    feature_fault, values_fault, vector_fault = model.transform(data_test_fault)

    # Difference of test data
    diff_normal = np.diff(data_test_normal,axis=0)
    diff_fault = np.diff(data_test_fault, axis=0)
    diff_normal = np.array(diff_normal)
    diff_fault = np.array(diff_fault)

    # Statistics of test data
    Te_train = np.diag(np.dot(feature_normal_train[:, :6], feature_normal_train[:, :6].transpose()))
    Td_train = np.diag(np.dot(feature_normal_train[:, 6:], feature_normal_train[:, 6:].transpose()))
    Te_normal = np.diag(np.dot(feature_normal[:,:6],feature_normal[:,:6].transpose()))
    Td_normal = np.diag(np.dot(feature_normal[:,6:],feature_normal[:,6:].transpose()))
    Te_fault = np.diag(np.dot(feature_fault[:,:6],feature_fault[:,:6].transpose()))
    Td_fault = np.diag(np.dot(feature_fault[:,6:],feature_fault[:,6:].transpose()))

    # Calculating Statistical Control Limits Using KDE Nonparametric Estimation
    confidence = 0.95
    Td_kzx = find_kde(Td_train, confidence)
    Te_kzx = find_kde(Te_train, confidence)

    # Splice the normal working condition of the test data with the fault
    Td_test = np.concatenate([Td_normal,Td_fault],axis=0)
    Te_test = np.concatenate([Te_normal,Te_fault],axis=0)

    # Visualization of statistical monitoring
    plt.suptitle("Test statistics and control limits")
    plt.subplot(2,1,1)
    plt.plot(Td_test)
    plt.ylabel('Td_test')
    plt.axhline(Td_kzx, c='r')
    plt.subplot(2,1,2)
    plt.plot(Te_test)
    plt.ylabel('Te_test')
    plt.axhline(Te_kzx, c='r')
    plt.show()







