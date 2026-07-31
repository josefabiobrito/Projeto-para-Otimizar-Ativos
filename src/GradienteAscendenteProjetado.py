import numpy as np
import time
from src.Otimizacao import Otimizacao
from src.Methods import busca_linear_backtracking 

class GradienteAscendenteProjetado(Otimizacao):
    """
    Implementa o método de Gradiente Projetado (Descida de Gradiente no Sharpe Negativo com Restrições).
    Se falhar, usa Gradiente Numérico com busca linear (Backtracking) e Projeção (Soma=1, Bounds).
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray, r_f: float):
        super().__init__(mu, cov, r_f)
        self.gerar_grad_hess()

    def fit(self, x0=None, alpha_0=None, rho=0.5, c=1e-4, max_iter=5000, 
            tol=1e-6, verbose=False):
        
        start = time.time()
        n = self.n
        
        # Inicialização
        if x0 is None:
            x_k = np.ones(n) / n
        else:
            x_k = np.array(x0, dtype=float)

        historico_x = [x_k.copy()]
        nit = 0

        func_obj = self._sharpe 
        grad_obj = self._grad_sharpe

        for k in range(max_iter):
            nit = k
            
            # Calcular Gradiente
            g_k = grad_obj(x_k)
            
            # Critério de Parada (Norma do Gradiente)
            if np.linalg.norm(g_k) < tol:
                if verbose: print("Convergiu por norma do gradiente.")
                break

            # Direção de Descida (para minimizar o Sharpe Negativo)
            p_k = -g_k 

            # Busca Linear (Backtracking)
            if alpha_0 is None:
                # Usa alpha inicial 1.0 ou adapta dinamicamente
                alpha_k = busca_linear_backtracking(
                    f=func_obj, 
                    grad_f=grad_obj, 
                    xk=x_k, 
                    pk=p_k, 
                    alpha_0=1.0, 
                    rho=rho, 
                    c=c
                )
            else:
                alpha_k = alpha_0

            # Passo de Atualização
            x_next = x_k + alpha_k * p_k

            # Traz o ponto de volta para o espaço viável
            x_next = self._proj_simplex(x_next)
            
            # Verificação de convergência por variação no x 
            if np.linalg.norm(x_next - x_k) < tol:
                x_k = x_next
                break

            x_k = x_next
            historico_x.append(x_k.copy())

            if verbose and k % 100 == 0:
                val = -func_obj(x_k) # Sharpe positivo para display
                print(f"Iter {k}: Sharpe={val:.4f} |Step={alpha_k:.2e}")
        
        historico_x.append(x_k.copy())
        end = time.time()
        
        # Sharpe final (lembrando que self._sharpe retorna negativo)
        sharpe_final = -self._sharpe(x_k)

        return {
            "method": "GradienteProjetado-Backtracking" if alpha_0 is None else "GradienteProjetado",
            "w": x_k,
            "sharpe": sharpe_final,
            "success": True, 
            "cost_final": -sharpe_final, 
            "time": end - start,
            "nit_total": nit,
            "history": historico_x
        }