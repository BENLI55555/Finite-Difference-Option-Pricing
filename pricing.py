from payoffs import *

def price_option(S0, E, r, sig, T, Nt, 
                 N1, N2, dx, 
                 omega = None, 
                 tol = None,
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None): 

    if option_style == 'european':
        V0 = european_payoff(S0, E, r, sig, T, Nt, N1, N2, dx,
                    option_type = option_type)[N1][-1]

    elif option_style == 'american':
        V0 = american_payoff(S0, E, r, sig, T, Nt, N1, N2, dx,
                    omega = omega, 
                    tol = tol,
                    option_type = option_type)[N1][-1]

    elif option_style == 'barrier':
        V0 = barrier_payoff(S0, E, r, sig, T, 
                    Nt = Nt, N1 = N1, N2 = N2, 
                    B = barrier, 
                    barrier_type = barrier_type, 
                    option_type = option_type)[N1][-1]

    return V0

