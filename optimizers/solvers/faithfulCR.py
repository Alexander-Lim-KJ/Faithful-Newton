# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:04:59 2024

@author: uqalim8
"""
import torch

def faithfulCR(A, b, term, maxit, skip_check = 1, reOrtho = True):
    k = 1
    p, r = b, b
    Ap = Avec(A, p)
    norm_b = torch.norm(b)
    
    # re-orthogonalization 
    if reOrtho:
        normAp = torch.norm(Ap)
        AP = Ap.reshape(-1, 1) / normAp
        
    Ar = Ap.clone()
    rAr = torch.dot(r, Ar)
    alpha = rAr / torch.dot(Ap, Ap)
    x = alpha * p
    
    d = 1 # number of times term is called
    # termination condition 
    if not term(x, 1):
        return x / 2, d, k, 0, "GRD"
    
    stored_dir = [(x, 1)]
    while True:            
        rp1 = r - alpha * Ap
        
        # re-orthogonalization 
        if reOrtho:
            rp1 = rp1 - AP @ (AP.T @ rp1) 
            
        Arp1 = Avec(A, rp1)
        rp1Arp1 = torch.dot(rp1, Arp1)
        beta = rp1Arp1 / rAr
        p = rp1 + beta * p
        Ap = Arp1 + beta * Ap
    
        if reOrtho:
            normAp = torch.norm(Ap)
            AP = torch.concat([AP, Ap.reshape(-1, 1) / normAp], dim = 1)
                
        # update
        Ar = Arp1
        rAr = rp1Arp1
        r = rp1
        k += 1
 
        alpha = rAr / torch.dot(Ap, Ap)
        x = x + alpha * p
        
        norm_rp1 = torch.norm(r)
        # only check for termination condition every skip_check times
        if not len(stored_dir) % skip_check and not term(x, (norm_b / norm_rp1) ** 2):
            d += 1
            return *binary_search(stored_dir, term, d), k, torch.norm(b - Avec(A, x)) / torch.norm(b), "TER"
        
        # clear stored update directions
        if not len(stored_dir) % skip_check:
            d += 1
            stored_dir = [(x, (norm_b / norm_rp1) ** 2)]
        else:
            stored_dir.append((x, (norm_b / norm_rp1) ** 2))
        
        # maximum iteration detection 
        if k >= maxit:
            return *binary_search(stored_dir, term, d), k, torch.norm(b - Avec(A, x)) / torch.norm(b), "MAX"
            
def binary_search(xs, term, d):
    t = len(xs)
    if t == 1:
        return xs[0][0], d 
    t, d = t // 2, d + 1
    if term(xs[t][0], xs[t][1]):
        return binary_search(xs[t:], term, d)
    return binary_search(xs[:t], term, d)

def faithfulCR_F(A, b, term, maxit, omega, reOrtho = True):
    k = 1
    p, r = b, b
    Ap = Avec(A, p)
    
    # re-orthogonalization 
    if reOrtho:
        normAp = torch.norm(Ap)
        AP = Ap.reshape(-1, 1) / normAp
        
    Ar = Ap.clone()
    rAr = torch.dot(r, Ar)
    alpha = rAr / torch.dot(Ap, Ap)
    x = alpha * p
    
    # NPC detection 
    if rAr <= 0:
        return b, k, "NPC(R)"
    
    # termination condition 
    if not term(x, k):
        return b, k, "GRD"
            
    while True:            
        rp1 = r - alpha * Ap
        
        # re-orthogonalization 
        if reOrtho:
            rp1 = rp1 - AP @ (AP.T @ rp1) 
            
        Arp1 = Avec(A, rp1)
        rp1Arp1 = torch.dot(rp1, Arp1)
        beta = rp1Arp1 / rAr
        p = rp1 + beta * p
        Ap = Arp1 + beta * Ap
    
        if reOrtho:
            normAp = torch.norm(Ap)
            AP = torch.concat([AP, Ap.reshape(-1, 1) / normAp], dim = 1)
                
        # update
        Ar = Arp1
        rAr = rp1Arp1
        r = rp1
        k += 1
 
        alpha = rAr / torch.dot(Ap, Ap)
        x = x + alpha * p
        
        # NPC detection 
        if rAr <= 0:
            if torch.norm(r) ** 2 > omega * torch.norm(b) ** 2:    
                return r, k, "NPC(R)"   
            
            if term(x, k):
                return x, k, "NPC(N)"
            else:
                return x - alpha * p, k, "NPC(N-1)"
        
        if not term(x, k):
            return x - alpha * p, k, "TER"
        
        # maximum iteration detection 
        if k >= maxit:
            return x, k, "MAX"

def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)

def resTerm(x, A, b, tol):
    return torch.norm(b - Avec(A, x)) / torch.norm(b) > tol

if "__main__" == __name__:
    torch.manual_seed(2024)
    N = 100
    MAXIT = 100
    D = torch.rand(N, dtype = torch.float64)
    A = torch.rand(N, N, dtype = torch.float64)
    A = (A.T + A) / 2
    _, eigV = torch.linalg.eigh(A)
    A = eigV.T @ torch.diag(D) @ eigV
    b = torch.rand(N, dtype = torch.float64)
    x, k, relr, dtype = faithfulCR(A, b, lambda x : resTerm(x, A, b, 1e-6), MAXIT, False)
