import matplotlib.pyplot as plt
import numpy as np
import warnings
import random

from src.BarreiraLogaritmicaPenalizacaoQuadratica import BarreiraLogaritmicaPenalizacaoQuadratica
from src.GradienteAscendenteProjetado import GradienteAscendenteProjetado
from src.GradienteEspectralProjetado import GradienteEspectralProjetado
from src.Dados import *

# ----------------------
# Preparação dos Dados
# ----------------------
def main():
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

    # -------------------------
    # Comparacao das Trajetorias
    # --------------------------


    tickers = ['UGPA3.SA', 'MULT3.SA', 'WEGE3.SA']
    cov_ = cov.loc[tickers, tickers].values
    mu_ = mu.loc[tickers]
    r_f = 0.05

    def sharpe_ratio(mu, cov, r_f):
        def f(w):
            w = np.array(w)
            Rw = w @ mu
            var = w @ cov @ w
            return (Rw - r_f) / np.sqrt(var) if var > 0 else 0
        return f

    f_sharpe = sharpe_ratio(mu_, cov_, r_f)

    N = 200
    w1 = np.linspace(0, 1, N)
    w2 = np.linspace(0, 1, N)
    W1, W2 = np.meshgrid(w1, w2)
    W3 = 1 - W1 - W2

    mask = (W1 >= 0) & (W2 >= 0) & (W3 >= 0)
    Sharpe = np.full_like(W1, np.nan)

    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                Sharpe[i, j] = f_sharpe([W1[i, j], W2[i, j], W3[i, j]])

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.get_cmap('viridis')
    c = ax.contourf(W1, W2, Sharpe, levels=25, cmap=cmap)
    fig.colorbar(c, ax=ax, shrink=0.8, label="Índice de Sharpe")

    contours = ax.contour(W1, W2, Sharpe, levels=12, colors='white', alpha=0.5, linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    ax.plot([0, 1], [1, 0], color='black', linewidth=1.5, linestyle='-')
    ax.plot([0, 0], [0, 1], color='black', linewidth=1.5, linestyle='-')
    ax.plot([0, 1], [0, 0], color='black', linewidth=1.5, linestyle='-')

    ax.set_xlabel(f"{tickers[0]}", fontsize=11)
    ax.set_ylabel(f"{tickers[1]}", fontsize=11)
    ax.set_title("Superfície do Índice de Sharpe no Simplex e Trajetórias", fontsize=12)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    lista_metodos = [
        (GradienteAscendenteProjetado, 'green', 'Gradiente Projetado', 's'),
        (BarreiraLogaritmicaPenalizacaoQuadratica, 'red', 'Barreira Log + Penalidade', '^'),
        (GradienteEspectralProjetado, 'blue', 'Gradiente Espectral Projetado', '*')
    ]

    for ClasseMetodo, cor, label, marker_fim in lista_metodos:
        print(f"Executando {label}...")
        metodo = ClasseMetodo(mu_, cov_, r_f)
        x0 = np.array([0.1, 0.1, 0.8]) 
        solucao = metodo.fit(x0=x0)
        historico_array = np.array(solucao['history'])
        caminho_w1 = historico_array[:, 0]
        caminho_w2 = historico_array[:, 1]
        n_passos = len(caminho_w1)

        ax.plot(caminho_w1, caminho_w2, 
                color=cor, linestyle='--', linewidth=1.0, alpha=0.6, label=label + f" - n={n_passos}")

        alphas = np.linspace(0.1, 0.9, n_passos)
        ax.scatter(caminho_w1[1:-1], caminho_w2[1:-1], 
                color=cor, s=10, alpha=alphas[1:-1], marker='o', zorder=3)

        ax.plot(caminho_w1[0], caminho_w2[0], 
                marker='o', color=cor, markersize=10, zorder=4) 
                
        ax.plot(caminho_w1[-1], caminho_w2[-1], 
                marker=marker_fim, color=cor, markersize=10, zorder=5, alpha=1.0)

    ax.legend(loc='upper right', frameon=True, fontsize=9)
    plt.show()

if __name__ == "__main__":
    main()