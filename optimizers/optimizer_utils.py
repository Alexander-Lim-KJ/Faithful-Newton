# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:01:15 2024

@author: uqalim8
"""
from .faithfulNewtonCG import FaithfulNewtonCG
from .faithfulNewtonCR import FaithfulNewtonCR
from .NewtonMR_NC import NewtonMR_NC
from .NewtonCR_NC import NewtonCR_NC
from .NewtonCG_NC import NewtonCG_NC
from .NewtonCG import NewtonCG
from .NewtonCR import NewtonCR

TEXT = "{:<20} : {:>20}"

def init_algorithms(fun, x0, algo, c):
    
    if algo == "FaithfulNewtonCR":
        print(TEXT.format("Algorithm", algo))
        return FaithfulNewtonCR(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.beta, c.skips, 
                                c.inmaxite, c.lineMaxite, c.lineBetaB, c.lineRho)
    
    if algo == "FaithfulNewtonCG":
        print(TEXT.format("Algorithm", algo))
        return FaithfulNewtonCG(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.beta, c.skips, 
                                c.inmaxite, c.lineMaxite, c.lineBetaB, c.lineRho)
    
    if algo == "NewtonCG":
        print(TEXT.format("Algorithm", algo))
        return NewtonCG(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                        c.lineMaxite, c.lineBeta, c.lineRho)
    
    if algo == "NewtonCR":
        print(TEXT.format("Algorithm", algo))
        return NewtonCR(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                        c.lineMaxite, c.lineBeta, c.lineRho)
        
    if algo == "NewtonMR-NC":
        print(TEXT.format("Algorithm", algo))
        return NewtonMR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                           c.lineMaxite, c.lineBetaB, c.lineRho, c.lineBetaFB)
    
    if algo == "NewtonCR-NC":
        print(TEXT.format("Algorithm", algo))
        return NewtonCR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                           c.lineMaxite, c.lineBetaB, c.lineRho, c.lineBetaFB)                   
    
    if algo == "NewtonCappedCG":
        print(TEXT.format("Algorithm", algo))
        return NewtonCG_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, c.restol, c.inmaxite, 
                           c.lineMaxite, c.lineBeta, c.lineRho, c.epsilon)
    
    # if algo == "Linesearch_GD":
    #     print(TEXT.format("Algorithm", algo))
    #     return optAlgs.linesearchGD(fun, x0, c.alpha0, c.gradtol, c.maxite, 
    #                                 c.maxorcs, c.lineMaxite, c.lineBetaB, c.lineRho)
    
    # if algo == "NewtonCG_NC_FW":
    #     print(TEXT.format("Algorithm", algo))
    #     return optAlgs.NewtonCG_NC_FW(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
    #                                c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
    #                                c.lineRho, c.epsilon, c.Hsub)
    
    # if algo == "NewtonCG_TR_Steihaug":
    #     print(TEXT.format("Algorithm", algo))
    #     return optAlgs.NewtonCG_TR_Steihaug(fun, x0, c.gradtol, c.maxite, c.maxorcs, 
    #                                         c.restol, c.inmaxite, c.deltaMax, c.delta0, 
    #                                         c.eta, c.eta1, c.eta2, c.gamma1, c.gamma2, c.Hsub)
    
    # if algo == "L-BFGS":
    #     print(TEXT.format("Algorithm", algo))
    #     return optAlgs.L_BFGS(fun, x0, c.alpha0, c.gradtol, c.m, 
    #                           c.maxite, c.maxorcs, c.lineMaxite)
