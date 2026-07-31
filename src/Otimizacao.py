import numpy as np
import time
import sympy as sp

class Otimizacao:
    def __init__(self, mu, cov, r_f):
        self.mu = np.asarray(mu, float)
        self.cov = np.asarray(cov, float)
        self.r_f = float(r_f)
        self.w = None
        self.n = self.mu.size
        self._sharpe_sym = None
        self._grad_sym = None
        self._hess_sym = None

    def fit(self):
        inicio = time.time()
        fim = time.time()
        self.time_last = fim - inicio

    def time(self, n=10):
        tempos = []
        for _ in range(n):
            inicio = time.time()
            self.fit()
            tempos.append(time.time() - inicio)
        return sum(tempos) / n
    
    
    def _proj_simplex(self, v):
        """
        Algoritmo de projeção no simplexo.
        Fornecido pelo usuário.
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w


    def gerar_grad_hess(self):
        mu = sp.Matrix(self.mu)
        cov = sp.Matrix(self.cov)
        n = self.n

        w = sp.symbols(f"w0:{n}", real=True)
        w_vec = sp.Matrix(w)

        var = (w_vec.T * cov * w_vec)[0]
        sharpe_neg = -(mu.dot(w_vec) - self.r_f) / sp.sqrt(var)

        grad_sym = [sp.diff(sharpe_neg, wi) for wi in w_vec]
        hess_sym = [[sp.diff(grad_sym[i], w_vec[j]) for j in range(n)] for i in range(n)]

        sharpe_l = sp.lambdify(w, sharpe_neg, "numpy")
        grad_l = sp.lambdify(w, grad_sym, "numpy")
        hess_l = sp.lambdify(w, hess_sym, "numpy")

        def sharpe_func(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != n:
                raise ValueError(f"sharpe_func espera vetor tamanho {n}, recebeu {x.size}")
            return float(sharpe_l(*x))

        def grad_func(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != n:
                raise ValueError(f"grad_func espera vetor tamanho {n}, recebeu {x.size}")
            g = np.array(grad_l(*x), dtype=float).reshape(-1)
            return g

        def hess_func(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != n:
                raise ValueError(f"hess_func espera vetor tamanho {n}, recebeu {x.size}")
            H = np.array(hess_l(*x), dtype=float)
            return H

        self._sharpe_sym = sharpe_func
        self._grad_sym = grad_func
        self._hess_sym = hess_func

        return self._sharpe_sym, self._grad_sym, self._hess_sym

    def _sharpe(self, w):
        if self._sharpe_sym is None:
            raise RuntimeError("Função simbólica não gerada — chame gerar_grad_hess() primeiro.")
        return float(self._sharpe_sym(w))

    def _grad_sharpe(self, w):
        if self._grad_sym is None:
            raise RuntimeError("Gradiente simbólico não gerado — chame gerar_grad_hess() primeiro.")
        return np.array(self._grad_sym(w), dtype=float).reshape(-1)

    def _hess_sharpe(self, w):
        if self._hess_sym is None:
            raise RuntimeError("Hessiana simbólica não gerada — chame gerar_grad_hess() primeiro.")
        return np.array(self._hess_sym(w), dtype=float)

