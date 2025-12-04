import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
from scipy.optimize import minimize


from src.BarreiraLogaritmicaPenalizacaoQuadratica import BarreiraLogaritmicaPenalizacaoQuadratica
from src.GradienteAscendenteProjetado import GradienteAscendenteProjetado
from src.GradienteEspectralProjetado import GradienteEspectralProjetado
from src.Dados import *

# ----------------------
# Preparação dos Dados
# ----------------------

random.seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

obter_ibxl()
acoes = pd.read_csv(CSV_PATH)['codigo']
dados = get_dados(acoes)
taxa_livre_risco = obter_selic_atual()
metricas = get_metricas(dados)
# r_f = obter_selic_atual()
r_f = 0.05
cov = metricas['covariancia']
mu = metricas['retorno_medio']
n_ativos = len(mu)

def extrair_nit(solucao):
    """Tenta extrair número de iterações de qualquer método."""
    try:
        return solucao['nit_total']
    except:
        return np.nan

def resolver_benchmark_scipy(mu, cov, r_f):
    """
    Resolve o problema usando scipy.optimize.minimize (SLSQP) 
    para servir de 'Ground Truth'.
    """
    n = len(mu)
    w0 = np.ones(n) / n
    
    # Função objetivo: Minimizar Sharpe Negativo
    def neg_sharpe(w):
        retorno = w @ mu
        risco = np.sqrt(w @ cov @ w)
        if risco < 1e-9: return 0.0
        return -(retorno - r_f) / risco

    # Restrições: Soma = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # Limites: 0 <= w <= 1
    bounds = tuple((0, 1) for _ in range(n))
    
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-9)
    
    sharpe_otimo = -res.fun
    return sharpe_otimo, res.x, res.success, res.nit

def rodar_metodo(MetodoCls, mu, cov, r_f, verbose=True):
    """Executa cada método e coleta tempo, nit e Sharpe."""

    nome_metodo = MetodoCls.__name__
    if verbose:
        print(f"\n--- Rodando metodo: {nome_metodo} ---")

    metodo = MetodoCls(mu, cov, r_f)

    # mede fit
    t0 = time.time()
    solucao = metodo.fit()
    t1 = time.time()

    if verbose:
        print(f"    Fit concluido em {t1 - t0:.4f}s")

    # mede time() interno
    try:
        tempo_medio = metodo.time(n=5)
        if verbose:
            print(f"    metodo.time(): {tempo_medio:.4f}s")
    except:
        tempo_medio = t1 - t0
        if verbose:
            print(f"    metodo.time() nao disponivel, usando tempo do fit.")

    nit = extrair_nit(solucao)
    if verbose:
        print(f"    Iteracoes: {nit}")

    return {
        "Sharpe": solucao.get("sharpe", np.nan),
        "Tempo (s)": tempo_medio,
        "Iterações": nit,
        "Success": solucao.get("success", True),
        "w": solucao.get("w", None)
    }

def comparar_metodos(mu, cov, r_f, verbose=True):
    """Roda todos os métodos para um conjunto de ativos e compara com Scipy."""
    metodos = {
        "SPG": GradienteEspectralProjetado,
        "Barreira + Penalizacao": BarreiraLogaritmicaPenalizacaoQuadratica,
        "Gradiente Projetado": GradienteAscendenteProjetado
    }

    if verbose:
        print("\n===================================")
        print(" Comparando metodos...")
        print("===================================")

    # 1. Calcular Ground Truth com Scipy
    if verbose: print("\n--- Calculando Benchmark (Scipy) ---")
    sharpe_ref, w_ref, success_ref, nit_ref = resolver_benchmark_scipy(mu, cov, r_f)
    
    if verbose:
        print(f"    Benchmark Sharpe: {sharpe_ref:.6f}")
        print(f"    Benchmark Success: {success_ref}")

    resultados = []

    # Adiciona a linha do Benchmark na tabela também
    resultados.append({
        "Método": "Benchmark (Scipy)",
        "Sharpe": sharpe_ref,
        "Gap Sharpe": 0.0,
        "Tempo (s)": np.nan, # Nao medimos tempo do scipy aqui
        "Iterações": nit_ref,
        "Success": success_ref,
        "w": w_ref
    })

    # 2. Rodar Métodos Customizados
    for nome, MetodoCls in metodos.items():
        try:
            res = rodar_metodo(MetodoCls, mu, cov, r_f, verbose=verbose)
            res["Método"] = nome
            
            # Calcula a diferença para o ótimo
            if not np.isnan(res["Sharpe"]):
                res["Gap Sharpe"] = abs(sharpe_ref - res["Sharpe"])
            else:
                res["Gap Sharpe"] = np.nan

        except Exception as e:
            if verbose:
                print(f"    ERRO no metodo {nome}: {e}")
            res = {
                "Método": nome,
                "Sharpe": np.nan,
                "Gap Sharpe": np.nan,
                "Tempo (s)": np.nan,
                "Iterações": np.nan,
                "Success": False,
                "Erro": str(e),
                "w": None
            }
        resultados.append(res)

    if verbose:
        print("\nTodos os metodos testados.")
    
    return pd.DataFrame(resultados)

def tabela_analise(metricas, r_f, verbose=True):
    """
    Seleciona amostras aleatórias de ativos (n=3,10,25)
    e gera tabela comparativa para cada tamanho.
    """

    acoes = list(metricas["retorno_medio"].index)
    tamanhos = [3, 10, 25]
    tabelas = {}

    if verbose:
        print("\n===================================")
        print(" Iniciando analise para tamanhos n = 3, 10, 25")
        print("===================================")

    for n in tamanhos:

        if verbose:
            print(f"\n\n==============================")
            print(f" Selecionando {n} ativos...")
            print("==============================")

        if n > len(acoes):
            if verbose:
                print(f"    Numero de ativos ({len(acoes)}) menor que {n}. Pulando.")
            continue

        tickers = np.random.choice(acoes, size=n, replace=False)
        if verbose:
            print(f"    Tickers selecionados: {list(tickers)}")

        mu_ = metricas["retorno_medio"].loc[tickers].values.flatten()
        cov_ = metricas["covariancia"].loc[tickers, tickers].values

        if verbose:
            print(f"    Rodando metodos para {n} ativos...")

        tabela = comparar_metodos(mu_, cov_, r_f, verbose=verbose)
        tabela["n_ativos"] = n
        tabela["Tickers"] = [list(tickers)] * len(tabela)

        if verbose:
            print(f"Concluido para n={n}")

        tabelas[n] = tabela

    if verbose:
        print("\n\nAnalise finalizada!")
    return tabelas

if __name__ == "__main__":
    res = tabela_analise(metricas, r_f=0.05)
    df_final = pd.concat(res.values(), ignore_index=True)
    df_final.to_csv('resultados.csv', index=False)