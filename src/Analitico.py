
import numpy as np
import time

from src.Otimizacao import Otimizacao


class Analitico(Otimizacao):
    def __init__(self, mu, cov, r_f):
        super().__init__(mu, cov, r_f)  
        self.gerar_grad_hess()
            

    def fit(self, tol=1e-6):
        start = time.time()
        n = self.n

        try:
            cov_inv = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(self.cov)
        
        excesso_retorno = self.mu - self.r_f
        pesos_analiticos = cov_inv @ excesso_retorno
        
        soma_analitica = np.sum(pesos_analiticos)
        if soma_analitica != 0:
            pesos_analiticos /= soma_analitica
        
        # Verifica se a solução analítica respeita os bounds 
        if (np.all(pesos_analiticos >=  - tol) and 
            np.all(pesos_analiticos <= 1 + tol) and 
            (np.abs(pesos_analiticos.sum() - 1) <= tol)):
            
            # Aplica o clip final para garantir exatidão
            pesos_analiticos = np.clip(pesos_analiticos, 0, 1)
            pesos_analiticos /= np.sum(pesos_analiticos)

            end = time.time()
            return {
                "method": "Analytical-Check",
                "w": pesos_analiticos,
                "sharpe": -self._sharpe(pesos_analiticos),
                "success": True,
                "time": end - start,
                "nit_total": 0,
                "history": [pesos_analiticos]
            }
        
        return {
            "method": "Analytical",
            "w": np.nan*pesos_analiticos,
            "sharpe": -self._sharpe(pesos_analiticos),
            "success": False,
            "time": 0,
            "nit_total": 0,
            "history": [pesos_analiticos]
        }