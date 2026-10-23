import numpy as np
from scipy import linalg
from solver import *

def european_payoff(S0, E, r, sig, T, Nt, 
                    N1 = None, N2 = None, dx = None,
                    option_type = 'call',
                    tri = None):
    if not tri:
        x0 = np.log(S0/E)
        xlow = x0 - N1 * dx
        xhigh = x0 + N2 * dx
        Nx = N1 + N2
    else:
        xlow, xhigh, Nx = tri
        dx = (xhigh - xlow)/Nx

    Tbar = 0.5 * sig ** 2 * T
    dt = Tbar/Nt
    k = 2 * r / sig ** 2
    alphaT = - (k - 1)/2
    beta = -(k + 1)**2/4

    alpha = dt/dx**2

    tau = np.linspace(0, Tbar, Nt + 1)
    x = np.linspace(xlow, xhigh, Nx + 1)

    if option_type == 'call':
        a = np.zeros(Nt + 1)
        b = np.exp((1 - alphaT) * xhigh - beta * tau) - np.exp(-alphaT * xhigh - (beta + k) * tau) 
        x0 = x[1:-1]
        u0 = np.maximum(np.exp((k + 1) * x0 / 2) - np.exp((k - 1) * x0 / 2), 0)
    elif option_type == 'put':
        a = np.exp(-alphaT * xlow - (beta + k) * tau)
        b = np.zeros(Nt + 1)
        x0 = x[1:-1]
        u0 = np.maximum(-np.exp((k + 1) * x0 / 2) + np.exp((k - 1) * x0 / 2), 0)        

    G = (1 + 2 * alpha) * np.ones(Nx - 1)
    subG = -alpha * np.ones(Nx - 2)
    superG = -alpha * np.ones(Nx - 2)

    U = np.zeros((Nx - 1, Nt + 1))
    P = np.zeros((Nx - 1, Nt + 1))
    P[0] = alpha * a
    P[-1] = alpha * b
    U[:, 0] = u0

    for i in range(1, Nt + 1):
        y = U[:, i - 1] + P[:, i]
        U[:, i] = thomas_algorithm(y, subG, G, superG)
    
    U = np.r_[[a], U, [b]]

    tt, xx = np.meshgrid(tau, x)
    V = E * np.multiply(np.exp(alphaT * xx + beta * tt), U)
    return V

def barrier_payoff(S0, E, r, sig, T, 
                    B, 
                    barrier_type, 
                    Nt = None, N1 = None, N2 = None, 
                    option_type = 'call',
                    tri = None):

    if not tri:
        if barrier_type == 'up_and_out' or barrier_type == 'up_and_in':
            x0 = np.log(S0/E)
            xhigh = np.log(B/E)
            dx = (xhigh - x0)/N2
            xlow = x0 - N1 * dx
        elif barrier_type == 'down_and_out' or barrier_type == 'down_and_in':
            x0 = np.log(S0/E)
            xlow = np.log(B/E)
            dx = (x0 - xlow)/N1
            xhigh = x0 + N2 * dx

        Nx = N1 + N2
    else:
        xlow, xhigh, Nx = tri
        dx = (xhigh - xlow)/Nx

    Tbar = 0.5 * sig ** 2 * T
    dt = Tbar/Nt
    k = 2 * r / sig ** 2
    alphaT = - (k - 1)/2
    beta = -(k + 1)**2/4

    alpha = dt/dx**2

    tau = np.linspace(0, Tbar, Nt + 1)
    x = np.linspace(xlow, xhigh, Nx + 1)

    if option_type == 'call':
        if barrier_type == 'up_and_out' or barrier_type == 'up_and_in':
            a = np.zeros(Nt + 1)
            b = np.zeros(Nt + 1)
        elif barrier_type == 'down_and_out' or barrier_type == 'down_and_in':
            a = np.zeros(Nt + 1)
            b = np.exp((1 - alphaT) * xhigh - beta * tau) - np.exp(-alphaT * xhigh - (beta + k) * tau) 
        x0 = x[1:-1]
        u0 = np.maximum(np.exp((k + 1) * x0 / 2) - np.exp((k - 1) * x0 / 2), 0)

    elif option_type == 'put':
        if barrier_type == 'up_and_out' or barrier_type == 'up_and_in':
            a = np.exp(-alphaT * xlow - (beta + k) * tau)
            b = np.zeros(Nt + 1)
        elif barrier_type == 'down_and_out' or barrier_type == 'down_and_in':
            a = np.zeros(Nt + 1)
            b = np.zeros(Nt + 1)
        x0 = x[1:-1]
        u0 = np.maximum(- np.exp((k + 1) * x0 / 2) + np.exp((k - 1) * x0 / 2), 0)       

    G = (1 + 2 * alpha) * np.eye(Nx - 1)
    subG = -alpha * np.diag(np.ones(Nx - 2), -1)
    superG = -alpha * np.diag(np.ones(Nx - 2), 1)
    G += (subG + superG)
    invG = linalg.inv(G)

    U = np.zeros((Nx - 1, Nt + 1))
    P = np.zeros((Nx - 1, Nt + 1))
    P[0] = alpha * a
    P[-1] = alpha * b
    U[:, 0] = u0

    for i in range(1, Nt + 1):
        U[:, i] = invG.dot(U[:, i - 1]) + invG.dot(P[:, i])
    
    U = np.r_[[a], U, [b]]

    tt, xx = np.meshgrid(tau, x)
    V = E * np.multiply(np.exp(alphaT * xx + beta * tt), U)

    if barrier_type == 'down_and_in':
        N0 = int(np.ceil((xlow + 10)/dx)) if xlow > -10 else 0
        X = european_payoff(S0, E, r, sig, T, Nt, N1 + N0, N2, dx, 
                            option_type=option_type)[N0:]
        V = X - V
    elif barrier_type == 'up_and_in':
        N3 = int(np.ceil((10 - xhigh)/dx)) if xhigh < 10 else 0
        X = european_payoff(S0, E, r, sig, T, Nt, N1, N2 + N3, dx, 
                            option_type=option_type)[: N1 + N2 + 1]        
        V = X - V   
    return V

def american_payoff(S0, E, r, sig, T, Nt, 
                    N1 = None, N2 = None, dx = None,
                    omega = 1.8, 
                    tol = 0.00001,
                    option_type = 'call',
                    tri = None):
    if not tri:
        x0 = np.log(S0/E)
        xlow = x0 - N1 * dx
        xhigh = x0 + N2 * dx
        Nx = N1 + N2
    else:
        xlow, xhigh, Nx = tri
        dx = (xhigh - xlow)/Nx

    if option_type == 'call':
        return european_payoff(S0, E, r, sig, T, Nt, N1, N2, dx)

    elif option_type == 'put':

        Tbar = 0.5 * sig ** 2 * T
        dt = Tbar/Nt
        k = 2 * r / sig ** 2
        alphaT = - (k - 1)/2
        beta = -(k + 1)**2/4

        alpha = dt/dx**2

        tau = np.linspace(0, Tbar, Nt + 1)
        x = np.linspace(xlow, xhigh, Nx + 1)

        a = np.exp(-alphaT * xlow - beta * tau)
        b = np.zeros(Nt + 1)
        x0 = x[1:-1]
        u0 = np.maximum(-np.exp((k + 1) * x0 / 2) + np.exp((k - 1) * x0 / 2), 0)   

        def g(x, tau, k):
            en = -np.exp((k + 1) * x / 2) + np.exp((k - 1) * x / 2)   
            ou = np.exp(- beta * tau)
            return ou * np.maximum(en, 0) 

        U = np.zeros((Nx - 1, Nt + 1))
        P = np.zeros((Nx - 1, Nt + 1))
        P[0] = alpha * a
        P[-1] = alpha * b
        U[:, 0] = u0

        for i in range(1, Nt + 1):
            u = np.zeros(Nx - 1)
            gee = g(x0, tau[i], k)
            d = U[:, i - 1] + P[:, i]
            convergence = False
            while convergence == False:
                old = u.copy()
                new = u.copy()
                for j in range(Nx - 1):
                    node = d[j]
                    if j > 0:
                        node += alpha * new[j - 1]
                    if j < Nx - 2:
                        node += alpha * new[j + 1]
                    node = (1 - omega) * old[j] + omega/(1 + 2 * alpha) * node
                    new[j] = max(node, gee[j])
                u = new
                if linalg.norm(u - old) < tol:
                    convergence = True
            U[:, i] = u
        
        U = np.r_[[a], U, [b]]

        tt, xx = np.meshgrid(tau, x)
        V = E * np.multiply(np.exp(alphaT * xx + beta * tt), U)
        return V


