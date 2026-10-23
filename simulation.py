from payoffs import *
import numpy as np

def simulate_gbw(S0, mu, sig, T, N, M, seed=None): #out:(M, N + 1)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, np.sqrt(T/N), size=(N, M))
    B = np.zeros((N+1, M))
    B[1:]=W.cumsum(axis=0)
    t = np.linspace(0,T,N+1).reshape(-1,1)
    S = S0*np.exp((mu-sig**2/2)*t+sig*B)
    
    return S.T

def simulate_s(S0, E, N1, N2, dx, 
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None):

    if option_style == 'european' or option_style == 'american':
        x0 = np.log(S0/E)
        xlow = x0 - N1 * dx
        xhigh = x0 + N2 * dx

    elif option_style == 'barrier':
        if barrier_type == 'up_and_out' or barrier_type == 'up_and_in':
            x0 = np.log(S0/E)
            xhigh = np.log(barrier/E)
            dx = (xhigh - x0)/N2
            xlow = x0 - N1 * dx

        elif barrier_type == 'down_and_out' or barrier_type == 'down_and_in':
            x0 = np.log(S0/E)
            xlow = np.log(barrier/E)
            dx = (x0 - xlow)/N1
            xhigh = x0 + N2 * dx

    Nx = N1 + N2
    s = E * np.exp(np.linspace(xlow, xhigh, Nx + 1))

    return s

def option_grid(S0, E, r, sig, T, Nt, 
                 N1, N2, dx, omega, tol,
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None): 

    if option_style == 'european':
        V = european_payoff(S0, E, r, sig, T, Nt, N1, N2, dx,
                    option_type = option_type)

    elif option_style == 'american':
        V = american_payoff(S0, E, r, sig, T, Nt, N1, N2, dx,
                    omega = omega, 
                    tol = tol,
                    option_type = option_type)

    elif option_style == 'barrier':
        V = barrier_payoff(S0, E, r, sig, T, 
                    Nt = Nt, N1 = N1, N2 = N2, 
                    B = barrier, 
                    barrier_type = barrier_type, 
                    option_type = option_type)

    return V

