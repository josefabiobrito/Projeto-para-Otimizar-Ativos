import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

def busca_linear_backtracking(f, grad_f, xk, pk, alpha_0, rho=0.5, c=1e-4, max_iter=50):
    alpha = alpha_0
    f_xk = f(xk)
    grad_fk = grad_f(xk)
    grad_dot_pk = np.dot(grad_fk, pk)
    if grad_dot_pk >= 0:
        raise ValueError("Direção pk não é de descida.")

    for _ in range(max_iter):
        if f(xk + alpha * pk) <= f_xk + c * alpha * grad_dot_pk:
            return alpha
        alpha *= rho
    return alpha

def gradient_descent(f, grad_f, x0, alpha_0=None, rho=0.5, c=1e-4, max_iter=1000, tol=1e-6, verbose=False):
    x_k = np.array(x0, dtype=float)
    historico_x = [x_k.copy()]
    for k in range(max_iter):
        g_k = grad_f(x_k)
        if np.linalg.norm(g_k) < tol:
            break
        p_k = -g_k
        if alpha_0 is None:
            alpha_k = busca_linear_backtracking(f, grad_f, x_k, p_k, 1.0, rho, c)
        else:
            alpha_k = alpha_0
        x_k = x_k + alpha_k * p_k
        historico_x.append(x_k.copy())
        if verbose:
            print(f"Iteração {k}: x = {x_k}, f(x) = {f(x_k):.4f}, ||grad|| = {np.linalg.norm(g_k):.4e}")
    return x_k, historico_x

def newton_method(f, grad_f, hess_f, x0, alpha_0=None, rho=0.5, c=1e-4, max_iter=50, max_reg_iter=10, mu_reg=1e-6, tol=1e-6, verbose=False):
    x_k = np.array(x0, dtype=float)
    historico_x = [x_k.copy()]
    for k in range(max_iter):
        g_k = grad_f(x_k)
        H_k = hess_f(x_k)

        if np.linalg.norm(g_k) < tol:
            break
        mu = 0.0 # Parâmetro de regularização (inicialmente zero)
        H_mod = H_k
        
        for reg_iter in range(max_reg_iter):
            try:
                C, lower = cho_factor(H_mod)
                p_k = cho_solve((C, lower), -g_k)
                break
            except np.linalg.LinAlgError:
                # Se falhar (não definida positiva), regulariza
                if mu == 0.0:
                    mu = mu_reg
                else:
                    mu *= 2.0 
                # Aplica a regularização
                H_mod = H_k + mu * np.eye(H_k.shape[0])

                if reg_iter == max_reg_iter - 1:
                    raise ValueError(f"Não foi possível obter uma Hessiana definida positiva após {max_reg_iter} tentativas na iteração {k}. Último mu: {mu}")
        if alpha_0 is None:
            alpha_k = busca_linear_backtracking(f, grad_f, x_k, p_k, 1.0, rho, c)
        else:
            alpha_k = alpha_0
        x_k = x_k + alpha_k * p_k
        historico_x.append(x_k.copy())
        if verbose:
            print(f"Iteração {k}: x = {x_k}, f(x) = {f(x_k):.4f}, ||grad|| = {np.linalg.norm(g_k):.4e}")
    return x_k, historico_x

if __name__ == '__main__':

    def f(x):
        return (x[0] - 1)**2 + 10 * (x[1] - 2)**2

    def grad_f(x):
        g1 = 2 * (x[0] - 1)
        g2 = 20 * (x[1] - 2)
        return np.array([g1, g2])

    def hess_f(x):
        return np.array([
            [2, 0],
            [0, 0.343]
        ])

    # Ponto inicial comum
    x0 = [0.0, 0.0]

    # Executar o Método de Gradiente
    x_opt, historico_x = gradient_descent(f, grad_f, x0, max_iter=50, tol=1e-6)
    print(x_opt, len(historico_x))
    # Executar o Método de Newton
    x_opt, historico_x = newton_method(f, grad_f, hess_f, x0, max_iter=50, tol=1e-6)
    print(x_opt, len(historico_x))
