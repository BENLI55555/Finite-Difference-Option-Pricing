from simulation import *
import numpy as np
from payoffs import *

def delta(S, E, r, sig, T, Nt, N1, N2,   #the time complexity will be O(Nt**2, N)
                 dx, omega, tol,          #the bias must be O(h*2)
                 option_type = 'call',      #the biggest dificulty is N points may not be possible to be in the same grid
                 option_style = 'european',  #but actually we can consider insert value between grid, but is less accurate
                 barrier = None,
                 barrier_type = None):

    de = []
    for i, s in enumerate(S):
        V = option_grid(s, E, r, sig, T, Nt, 
                    N1, N2, dx, omega, tol,
                    option_type = option_type,
                    option_style = option_style,
                    barrier = barrier,
                    barrier_type = barrier_type)

        V = V[:, ::-1]
        Vminu, V, Vplus = V[N1 - 1][i], V[N1][i], V[N1 + 1][i]
        
        Sgrid = simulate_s(s, E, N1, N2, dx,
                    option_style=option_style,
                    barrier=barrier,
                    barrier_type=barrier_type)

        hplus = Sgrid[N1 + 1] - Sgrid[N1]
        hminu = Sgrid[N1] - Sgrid[N1 - 1]

        co1 = -hplus/(hminu* (hplus + hminu)) 
        co2 = (hplus - hminu)/(hplus * hminu)
        co3 = hminu/(hplus* (hplus + hminu))

        x = co1 * Vminu + co2 * V + co3 * Vplus
        de.append(x)

    return de

def gamma(S, E, r, sig, T, Nt, N1, N2,   
                 dx, omega, tol,  
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None):

    ga = []
    for i, s in enumerate(S):
        V = option_grid(s, E, r, sig, T, Nt, 
                    N1, N2, dx, omega, tol,
                    option_type = option_type,
                    option_style = option_style,
                    barrier = barrier,
                    barrier_type = barrier_type)

        V = V[:, ::-1]
        Vminu, V, Vplus = V[N1 - 1][i], V[N1][i], V[N1 + 1][i]
        
        Sgrid = simulate_s(s, E, N1, N2, dx,
                    option_style=option_style,
                    barrier=barrier,
                    barrier_type=barrier_type)

        hplus = Sgrid[N1 + 1] - Sgrid[N1]
        hminu = Sgrid[N1] - Sgrid[N1 - 1]

        co1 = 2 / (hminu * (hplus + hminu))
        co2 = - 2/(hminu * hplus)
        co3 = 2 /(hplus*(hplus + hminu))

        x = co1 * Vminu + co2 * V + co3 * Vplus
        ga.append(x)

    return ga

def theta(S, E, r, sig, T, Nt, N1, N2,   #calculate the theta value along the stock price S's trajectory
                 dx, omega, tol,      #actually we can reduce the time complexity
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None):

    dt = T/Nt
    th = []
    for i, s in enumerate(S):
        V = option_grid(s, E, r, sig, T, Nt, N1, N2, dx, omega, tol,
                        option_type=option_type,
                        option_style=option_style,
                        barrier=barrier,
                        barrier_type=barrier_type)
        V = V[:, ::-1]
        if i == 0:
            x = (V[N1][1] - V[N1][0])/dt
        elif i == Nt:
            x = (V[N1][-1] - V[N1][-2])/dt
        else:
            x = (V[N1][i + 1] - V[N1][i - 1])/(2 * dt)
        th.append(x)

    return th

def vega(S, E, r, sig, T, h, Nt, N1, N2,   #h is the change of sigma
                 dx, omega, tol,  
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None):

    ve = []
    for i, s in enumerate(S):
        Vplus = option_grid(s, E, r, sig + h, T, Nt, N1, N2, dx, omega, tol,
                        option_type=option_type,
                        option_style=option_style,
                        barrier=barrier,
                        barrier_type=barrier_type)
        
        Vminu = option_grid(s, E, r, sig - h, T, Nt, N1, N2, dx, omega, tol,
                        option_type=option_type,
                        option_style=option_style,
                        barrier=barrier,
                        barrier_type=barrier_type)    

        Vplus = Vplus[:, ::-1]
        Vminu = Vminu[:, ::-1]

        x = (Vplus[N1][i] - Vminu[N1][i])/(2 * h)
        ve.append(x)

    return ve

def rho(S, E, r, sig, T, h, Nt, N1, N2,   #h is the change of r
                 dx, omega, tol,  
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None):

    rh = []
    for i, s in enumerate(S):
        Vplus = option_grid(s, E, r + h, sig, T, Nt, N1, N2, dx, omega, tol,
                        option_type=option_type,
                        option_style=option_style,
                        barrier=barrier,
                        barrier_type=barrier_type)
        
        Vminu = option_grid(s, E, r - h, sig, T, Nt, N1, N2, dx, omega, tol,
                        option_type=option_type,
                        option_style=option_style,
                        barrier=barrier,
                        barrier_type=barrier_type)    

        Vplus = Vplus[:, ::-1]
        Vminu = Vminu[:, ::-1]

        x = (Vplus[N1][i] - Vminu[N1][i])/(2 * h)
        rh.append(x)

    return rh
    


        


    

    
    
    

    
