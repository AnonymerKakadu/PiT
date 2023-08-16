'''
    File name: train_statistics.py
    Author: Richard Dirauf
    Python Version: 3.8
    Description: Helper program to read and visualize the trainings loss from the log file.
'''

import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fau_colors import cmaps

import sys

sys.path.append("..")
from gltr_error import GLTRError
colors = cmaps.fau


def main():
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",
                        type=str,
                        required=True,
                        help="Path to the log file.")
    args = parser.parse_args()

    # check values
    log_file = os.path.abspath(args.log)
    extracted_infos = []

    if not os.path.isfile(log_file):
        raise GLTRError('train_statistics', 'The log does not exist.')

    # open and go through the file
    epoch = 0
    loggy = open(log_file, "r")
    for line in loggy:
        if 'Finish' in line and 'Acc:' in line:
            # found line with values, extract them
            parts = line.split(' ')  # 7 parts
            epoch = int(parts[1])
            if epoch >= 501:
                break

            # add epoch, loss, acc to list
            extracted_infos.append([int(parts[1]), float(parts[4][:-1]), float(parts[6][:-1])])

    extracted_infos = np.array(extracted_infos)

    # plot loss and acc
    _, ax = plt.subplots()
    plt.ylabel('Loss')
    plt.xlabel("Epochs")
    plt.grid()
    plt.plot(extracted_infos[:, 0], extracted_infos[:, 1], color=colors[0])
    ax.set_xticks(range(0, epoch + 1, 50))
    ax.set_ylim([0, 2.5])
    ax.set_xlim([0, epoch])
    plt.show()

    _, ax = plt.subplots()
    plt.ylabel('Accuracy')
    plt.xlabel("Epochs")
    plt.grid()
    plt.plot(extracted_infos[:, 0], extracted_infos[:, 2], color=colors[0])
    ax.set_xticks(range(0, epoch + 1, 50))
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, epoch])
    plt.show()

    # save data
    ex_file = pd.DataFrame(extracted_infos, columns=['epoch', 'loss', 'acc'])
    ex_file.to_csv('training_statistics.csv', index=False)


if __name__ == "__main__":
    main()
