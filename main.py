# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

from utils import execute, saveRecords
import hyperparameters as hyper 
import sys

SEED = 1234 # integers
FOLDER_PATH = f"./{sys.argv[1]}"
PROBLEM_TYPE = "softmax_2-norm_CIFAR10" # functions: [softmax, FFN-MSELoss], 
                                        # regularizers: [None, non-convex, 2-norm, LASSO], 
                                        # datasets: [MNIST, MNISTb, MNISTsb, MNISTs, CIFAR10b, CIFAR10, Covtype]
LAMBDA = 0.0001
INITX0 = "zeros" #zeros, ones, uniform, normal, torch
VERBOSE = 1 #

if sys.argv[2] == "NewtonMR-NC":
    ALG = ("NewtonMR-NC", hyper.cMCRNPC) #("NewtonCG", hyper.cCG_NC)
elif sys.argv[2] == "NewtonCR-NC":
    ALG = ("NewtonCR-NC", hyper.cMCRNPC) #("NewtonCG", hyper.cCG_NC) 
elif sys.argv[2] == "NewtonCR":
    ALG = ("NewtonCR", hyper.cNWL)
elif sys.argv[2] == "NewtonCG":
    ALG = ("NewtonCG", hyper.cNWL)
elif sys.argv[2] == "FaithfulNewtonCR":
    ALG = ("FaithfulNewtonCR", hyper.cFN)
elif sys.argv[2] == "FaithfulNewtonCG":
    ALG = ("FaithfulNewtonCG", hyper.cFN)

if __name__ == "__main__":
    print("\n" + 21 * "**" + "\n" + 15 * " " + "Running on", str(hyper.cCUDA), "\n" + 21 * "**")
    algo = execute(FOLDER_PATH, PROBLEM_TYPE, LAMBDA, ALG[0], INITX0, ALG[1], VERBOSE, SEED)
    saveRecords(FOLDER_PATH, ALG[0], algo.record)
