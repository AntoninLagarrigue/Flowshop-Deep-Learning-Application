import unicodedata
import re
import numpy as np
import os
import io
import time
import math

# Functions for parsing the file


def saveLinewithC(line):
        """
        Supplement function to build the different sets of the seq2seq model.

        :param line: one line of the csv file
        :return: ptimes_list, solved_list.

        ptimes_list: the sequence with processing time and completion time.
        solved_list: the binary sequence that can indicate the window.
        """
        ptimes = line[0].split(' ')
        ptimes_list = []
        for k in range(1, len(ptimes), 4):
            ptimes_list.append(
                [int(float(ptimes[k])), int(float(ptimes[k + 1])), int(float(ptimes[k + 2])),
                 int(float(ptimes[k + 3]))])
        solved_list = list(map(int, line[1]))

        return ptimes_list, solved_list
        
        

def divideDatawithC(size, num_instance):
        """
        Function can divide the data into 3 part: Training set, Test set and Validation set.

        :param txtfile: the path of the database txt file.
        :param size: size of the sequence
        :return: X_train, y_train, X_test, y_test, X_validation, y_validation
        X_train: Training set which have the initial sequence with processing time and completion time.
        y_train: Training set which have the binary sequnece that can indicate the window.
        X_test: Test set which have the initial sequence with processing time and completion time.
        y_test: Test set which have the binary sequnece that can indicate the window.
        X_validation: Validation set which have the initial sequence with processing time and completion time.
        y_validation: Validation set which have the binary sequnece that can indicate the window.

        """

        print("num_instance: " + str(num_instance))
        num_ins_test = int(num_instance * 0.2)
        num_ins_validation = num_ins_test
        num_ins_train = num_instance - num_ins_test * 2
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        X_validation = []
        y_validation = []
        with open(path + 'databaseC.csv') as data:
            reader = csv.reader(data)
            dataSet = list(reader)
            length = len(dataSet)
            count = 0
            for line in dataSet:
                if len(line) == 1:
                    line_per_instance = 0
                    count = count + 1
                    continue
                line_per_instance += 1
                if line_per_instance < 5: # limit number of sequences per instances to 5
                  if count <= num_ins_train:
                      ptimes_list, solved_list = saveLinewithC(line)
                      X_train.append(ptimes_list)
                      y_train.append(solved_list)
                  if num_ins_train < count <= num_ins_train + num_ins_test:
                      ptimes_list, solved_list = saveLinewithC(line)
                      X_test.append(ptimes_list)
                      y_test.append(solved_list)
                  if num_ins_train + num_ins_test < count <= num_instance:
                      ptimes_list, solved_list = saveLinewithC(line)
                      X_validation.append(ptimes_list)
                      y_validation.append(solved_list)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        y_train = np.reshape(y_train, (len(y_train), size, 1))
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        y_test = np.reshape(y_test, (len(y_test), size, 1))
        X_validation = np.asarray(X_validation)
        y_validation = np.asarray(y_validation)
        y_validation = np.reshape(y_validation, (len(y_validation), size, 1))
        return X_train, y_train, X_test, y_test, X_validation, y_validation

