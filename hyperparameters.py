# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:10 2023

@author: uqalim8
"""
import torch

cTYPE = torch.float64
cCUDA = True

if cCUDA:
    cCUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    cCUDA = torch.device("cpu")

class const():
    pass

# Faithful-Newton
# Example: FaithfulNewtonCG, FaithfulNewtonCR
cFN = const()
cFN.alpha0 = 1
cFN.gradtol = 1e-6
cFN.maxite = 1e6
cFN.inmaxite = 5000
cFN.beta = 1e-4
cFN.skips = 1
cFN.maxorcs = 1e6
cFN.lineMaxite = 100
cFN.lineBetaB = 1e-4
cFN.lineRho = 0.5

# Newton's method with line-search
# Examples: NewtonCG, NewtonCR and NewtonMR
cNWL = const()
cNWL.alpha0 = 1
cNWL.gradtol = 1e-6
cNWL.maxite = 1e6
cNWL.maxorcs = 1e6
cNWL.inmaxite = 1000
cNWL.restol = 0.1
cNWL.lineMaxite = 100
cNWL.lineBeta = 1e-4
cNWL.lineRho = 0.5

# Stephen Wright's NewtonCappedCG
cCCG = const()
cCCG.alpha0 = 1
cCCG.gradtol = 1e-6
cCCG.maxite = 1e6
cCCG.maxorcs = 1e6
cCCG.restol = 1e-4
cCCG.inmaxite = 100
cCCG.lineMaxite = 100
cCCG.lineBeta = 1e-4
cCCG.lineRho = 0.5
cCCG.epsilon = 0.99

# Newton's method with NPC detection and forward / backward linesearch
cMCRNPC = const()
cMCRNPC.alpha0 = 1
cMCRNPC.gradtol = 1e-6
cMCRNPC.maxite = 1e6
cMCRNPC.restol = 10
cMCRNPC.inmaxite = 1000
cMCRNPC.maxorcs = 1e6
cMCRNPC.lineMaxite = 100
cMCRNPC.lineBetaB = 1e-4
cMCRNPC.lineRho = 0.5
cMCRNPC.lineBetaFB = 1e-4

cGD = const()
cGD.alpha0 = 1
cGD.gradtol = 1e-5
cGD.maxite = 1e5
cGD.maxorcs = 1e5
cGD.lineMaxite = 1000
cGD.lineBetaB = 1e-4
cGD.lineRho = 0.9

cTR_STEI = const()
cTR_STEI.gradtol = 1e-9
cTR_STEI.maxite = 1e6
cTR_STEI.inmaxite = 1000
cTR_STEI.maxorcs = 1e6
cTR_STEI.restol = 0.01                          
cTR_STEI.deltaMax = 1e10
cTR_STEI.delta0 = 1e5
cTR_STEI.eta = 0.01
cTR_STEI.eta1 = 1/4
cTR_STEI.eta2 = 3/4
cTR_STEI.gamma1 = 1/4
cTR_STEI.gamma2 = 2

cL_BFGS = const()
cL_BFGS.alpha0 = 1
cL_BFGS.gradtol = 1e-9
cL_BFGS.m = 20
cL_BFGS.maxite = 1e6
cL_BFGS.maxorcs = 1e6
cL_BFGS.lineMaxite = 100

cADAM = const()
cADAM.alpha0 = 0.00001
cADAM.beta1 = 0.9 #0.9
cADAM.beta2 = 0.999
cADAM.epsilon = 1e-8
cADAM.gradtol = 1e-9
cADAM.maxite = 1e8
cADAM.maxorcs = 1e6

cSGD = const()
cSGD.alpha0 = 0.5
cSGD.gradtol = 1e-5
cSGD.maxite = 1e8
cSGD.maxorcs = 1e5
