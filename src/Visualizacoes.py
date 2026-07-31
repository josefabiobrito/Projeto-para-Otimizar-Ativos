import matplotlib.pyplot as plt
import seaborn as sns


def graficoAcoes(df):
    ax = df.plot(figsize=(15,10))
    ax.set_title("Série histórica dos ativos")
    ax.set_ylabel("Preço (R$)")
    ax.set_xlabel("Data")
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return ax

def graficoRetorno(metricas):
    ax = metricas['retornos'].plot(figsize=(15, 10), alpha=0.7)
    ax.set_title("Volatilidade dos Retornos Diários", fontsize=14)
    ax.set_ylabel("Retorno Diário", fontsize=12)
    ax.set_xlabel("Data", fontsize=12)

    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', ncol=2)
    return ax

def graficoRAcumulado(metricas):
    retorno_acumulado = (1 + metricas['retornos']).cumprod() - 1

    ax = retorno_acumulado.plot(figsize=(15, 10), linewidth=2)

    ax.set_title("Trajetória de Retorno Acumulado", fontsize=14)
    ax.set_ylabel("Retorno Acumulado (%)", fontsize=12)
    ax.set_xlabel("Data", fontsize=12)

    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    return ax

def graficoVolatilidade(metricas):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = sns.barplot(x=metricas['volatilidade'].index, y=metricas['volatilidade'])
    ax.set_title("Comparativo de Volatilidade das Ações", fontsize=14, fontweight='bold')
    ax.set_ylabel("Volatilidade (Desvio Padrão)", fontsize=12)
    ax.set_xlabel("Ticker", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_correlacao(df_correlacao, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(df_correlacao,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                linewidths=.5,
                square=True,
                cbar_kws={"shrink": .8},
                ax=ax)

    ax.set_title("Matriz de Correlação", fontsize=14, pad=20)

    return ax

    