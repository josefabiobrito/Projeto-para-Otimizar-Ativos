import numpy as np
import time
from src.Otimizacao import Otimizacao

from src.Methods import busca_linear_backtracking

class GradienteEspectralProjetado(Otimizacao):
    """
    Implementa o Método do Gradiente Espectral Projetado (SPG - Spectral Projected Gradient).
    Utiliza o passo espectral de Barzilai-Borwein e busca linear de Armijo.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray, r_f: float):
        super().__init__(mu, cov, r_f)
        self.gerar_grad_hess()

        
    def fit(self, x0=None, lambda_min=1e-3, lambda_max=1e3, 
            rho=0.5, c=1e-4, max_iter=5000, tol=1e-6, verbose=False):
        """
        Executa o algoritmo SPG.
        
        Args:
            x0: Ponto inicial.
            lambda_min: Limite inferior para o passo espectral.
            lambda_max: Limite superior para o passo espectral.
            rho: Fator de redução da busca linear (rho1/rho2 simplificado).
            c: Parâmetro da condição de Armijo.
            tol: Tolerância para o critério de parada (estacionariedade).
        """
        
        start = time.time()
        n = self.n
        
        # Inicialização
        if x0 is None:
            x_k = np.ones(n) / n
        else:
            x_k = np.array(x0, dtype=float)
            x_k = self._proj_simplex(x_k)

        func_obj = self._sharpe      # Retorna Sharpe Negativo (Minimização)
        grad_obj = self._grad_sharpe # Gradiente do Sharpe Negativo

        # Avaliações iniciais
        f_k = func_obj(x_k)
        g_k = grad_obj(x_k)

        historico_x = [x_k.copy()]
        
        # Inicializa lambda (Arbitrário para a 1ª iteração, pois não temos s_k e y_k)
        lambda_k = 1.0 
        
        # Variáveis para guardar o passo anterior (k-1)
        x_old = x_k.copy()
        g_old = g_k.copy()

        for k in range(max_iter):
            # Critério de Parada 
            # Medida de estacionariedade: || P(x_k - g_k) - x_k || <= epsilon
            # Isso verifica se o gradiente projetado é nulo (ponto estacionário)
            pg_step = self._proj_simplex(x_k - g_k) - x_k
            norm_pg = np.linalg.norm(pg_step)
            
            if norm_pg <= tol:
                if verbose: print(f"Convergência alcançada na iteração {k}. Norma PG: {norm_pg:.2e}")
                break

            # --- Cálculo do Lambda ---
            if k > 0:
                s_k = x_k - x_old
                y_k = g_k - g_old
                
                sty = np.dot(s_k, y_k)
                sts = np.dot(s_k, s_k)
                
                # Garantir convergência
                if sty <= 0:
                    lambda_k = lambda_max
                else:
                    lambda_val = sts / sty
                    lambda_k = min(lambda_max, max(lambda_min, lambda_val))

            # Direção de Busca pk 
            w_k = x_k - lambda_k * g_k
            x_trial = self._proj_simplex(w_k)
            p_k = x_trial - x_k # Direção de descida viável
            
            alpha_k = busca_linear_backtracking(
                f=func_obj, 
                grad_f=grad_obj, 
                xk=x_k, 
                pk=p_k, 
                alpha_0=1.0, 
                rho=rho, 
                c=c
            )

            # Atualiza os parâmetros para o próximo loop
            x_new = x_k + alpha_k * p_k
            f_new = func_obj(x_new)

            x_old = x_k.copy()
            g_old = g_k.copy()
            
            x_k = x_new
            f_k = f_new
            g_k = grad_obj(x_k)
            historico_x.append(x_k.copy())
            
            if verbose and k % 100 == 0:
                # Exibimos -f_k para mostrar o Sharpe Positivo
                print(f"Iter {k}: Sharpe={-f_k:.4f} | Lambda={lambda_k:.2e} | NormPG={norm_pg:.2e}")

        end = time.time()
        sharpe_final = -func_obj(x_k)

        return {
            "method": "SPG (Spectral Projected Gradient)",
            "w": x_k,
            "sharpe": sharpe_final,
            "success": norm_pg <= tol, 
            "cost_final": f_k, 
            "time": end - start,
            "nit_total": k,
            "history": historico_x
        }