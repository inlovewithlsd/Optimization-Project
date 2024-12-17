import scipy
import numpy as np
from scipy import linalg

def calc_gamma_DR(
    x,
    f,
    direction,
    gap,
    lipschitz_t,
    eta=0.9,
    tau=2.0,
    max_step_size=1,
    max_iter=200):

    Qt = lambda gamma, M: - gamma*gap + (gamma**2 * M / 2) * linalg.norm(direction)**2
    f_t = f(x)

    lipschitz_t = eta * lipschitz_t
    if (lipschitz_t * linalg.norm(direction)**2) != 0:
        gamma_t = min(gap / (lipschitz_t * linalg.norm(direction)**2), 1)
    else:
        gamma_t = 1
     
    for _ in range(max_iter):
        if f(x + gamma_t * direction) - f_t <=  Qt(gamma_t, lipschitz_t):
            break
        else:
            lipschitz_t *= tau

    return gamma_t, lipschitz_t

def calc_gamma_adaptive(
    x,
    f_grad,
    direction,
    gap,
    lipschitz_t,
    eta=0.9,
    tau=2.0,
    max_step_size=1,
    max_iter=200):

    lipschitz_t = eta * lipschitz_t
    
    for _ in range(max_iter):
        gamma_t = min(gap / (lipschitz_t * linalg.norm(direction)**2), 1)
        if np.dot(direction, -f_grad(x + gamma_t*direction)) >= 0:
            break
        else:
            lipschitz_t *= tau

    return gamma_t, lipschitz_t

def calc_gamma_search(
    x,
    f,
    direction,
    gamma_t):

    eaxt_line_search = lambda gamma: f(x + gamma*direction)
    gamma = scipy.optimize.minimize(eaxt_line_search, gamma_t, bounds=[(0, 1)], method='L-BFGS-B', options={'ftol': 1e-8, 'gtol': 1e-8}).x[0]

    return gamma

    
    


def FW(
        x0,
        calc_f,
        calc_grad,
        calc_lmo,
        method,
        lipschitz=None,
        n_iter=1000,
        x_sol=None
    ):

    logs = {}

    x = x0.copy()
    lipschitz_t = None
    old_f_t = None

    start = perf_counter()
    
    for t in tqdm(range(1, n_iter+1)):
        grad = calc_grad(x)
        s = calc_lmo(grad, x)
        direction = s-x
        gap = np.dot(direction, -grad)
        
        if lipschitz_t is None:
            eps = 1e-3
            lipschitz_t = linalg.norm(grad - calc_grad(x + eps * direction)) / (eps * linalg.norm(direction) )

        if method == 'backtracing':
            gamma_t, lipschitz_t = calc_gamma_DR(x, calc_f, direction, gap, lipschitz_t)
        elif method == 'adaptive':
            gamma_t, lipschitz_t = calc_gamma_adaptive(x, calc_grad, direction, gap, lipschitz_t)
        elif method == 'line-search':
            gamma_t = calc_gamma_search(x, calc_f, direction, 2 / (t + 2))
        elif method == 'DR':
            assert lipschitz is not None
            gamma_t = min(gap / (lipschitz * linalg.norm(direction))**2, 1)
        else:
            gamma_t = 2 / (t + 2)
        
        logs[t] = {
            'f_x': calc_f(x),
            'fw': grad @ -direction,
            'dual': grad @ (x - x_sol),
            'primal': calc_f(x) - calc_f(x_sol),
            'gamma': gamma_t
        }
        
        x += gamma_t * direction

    return x, logs