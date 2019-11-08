"""
Functions to load the dataset.
"""
import numpy as np


def read_data(file_name):
    """This function is adapted from:
    https://github.com/benhamner/BioResponse/blob/master/Benchmarks/csv_io.py
    """
    f = open(file_name)
    # skip header
    f.readline()
    samples = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)
    return samples


def load():
    """Conveninence function to load all data as numpy arrays.
    """
    print("Loading data...")
    train = read_data("data/train.csv")
    y_train = np.array([x[0] for x in train])
    X_train = np.array([x[1:] for x in train])
    X_test = np.array(read_data("data/test.csv"))
    return X_train, y_train, X_test

def generate_data():
    """Conveninence function to load all data as numpy arrays.
    """
    print("Loading data...")
    train = read_data("data/train.csv")
    N = 1000
    D = 5
    np.random.seed(0)
    y_train = np.random.randint(0, 2, size=N)
    X_train = np.random.random((N, D))
    X_test = np.random.random((N//2, D))
    #X_train = np.array([x[1:] for x in train])
    #X_test = np.array(read_data("data/test.csv"))
    return X_train, y_train, X_test

if __name__ == '__main__':

    X_train, y_train, X_test = load()