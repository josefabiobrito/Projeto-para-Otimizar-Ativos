import yfinance as yf
import streamlit as st
import pandas as pd
import lxml
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
pd.set_option("display.min_rows",50)
#================================================
# FUNÇÕES DE OBTENÇÃO E OTIMIZAÇÃO DO PORTFÓLIO
#================================================

def get_dados(tickers,periodo = '4y'):
  tickers_b3 = [t+'.SA' if not t.endswith('.SA') else t for t in tickers]
  dados = yf.download(tickers_b3, period=periodo, interval='1d')
  precos = dados['Close']
  return precos

def get_metricas(dados_acoes):
  retornos = np.log(dados_acoes/dados_acoes.shift(1))
  retorno_medio = retornos.mean()*252
  covariancia = retornos.cov()*252
  variancia = retornos.var()*252
  volatilidade = retornos.std()*np.sqrt(252)
  correlacao = retornos.corr()
  metricas = {
    'retornos':retornos,
    'correlacao':correlacao,
    'retorno_medio':retorno_medio,
    'covariancia':covariancia,
    'variancia':variancia,
    'volatilidade':volatilidade
  }
  return metricas

#OTIMIZAÇÃO UTILIZANDO MÉTODO DO LAGRANGIANO E GRADIENTE DESCENTE
def otimizacao(metricas, r_alvo):
    taxa_livre_risco = obter_selic_atual()
    covariancia = metricas['covariancia']
    retorno_medio = metricas['retorno_medio']
    n_ativos = len(retorno_medio)

    try:
        cov_inv = np.linalg.inv(covariancia)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(covariancia)

    vetor_uns = np.ones_like(retorno_medio)

    A = retorno_medio.T @ cov_inv @ retorno_medio
    B = retorno_medio.T @ cov_inv @ vetor_uns
    C = vetor_uns.T @ cov_inv @ vetor_uns
    D = A * C - B * B

    if D <= 0:
        raise ValueError(f"Discriminante D={D:.6f} não é positivo!")

    ret_min = retorno_medio.min()
    ret_max = retorno_medio.max()

    if r_alvo < ret_min:
        r_alvo = ret_min
    elif r_alvo > ret_max:
        r_alvo = ret_max

    l1 = (C * r_alvo - B) / D
    l2 = (A - B * r_alvo) / D

    pesos_lagrange = cov_inv @ (l1 * retorno_medio + l2 * vetor_uns)

    tem_negativos = np.any(pesos_lagrange < 0)
    tem_maiores_que_1 = np.any(pesos_lagrange > 1)

    if tem_negativos or tem_maiores_que_1:

        pesos = np.array([1/n_ativos] * n_ativos)
        taxa_aprendizado = 0.1
        max_iter = 10000

        for iteracao in range(max_iter):
            pesos = np.clip(pesos, 0, 1)
            soma = np.sum(pesos)
            if soma > 0:
                pesos = pesos / soma

            ret_atual = pesos @ retorno_medio
            vol_atual = np.sqrt(pesos @ covariancia @ pesos)

            grad_risco = (covariancia @ pesos) / vol_atual if vol_atual > 1e-10 else covariancia @ pesos

            erro_retorno = ret_atual - r_alvo
            penalidade_retorno = 1000.0
            grad_retorno = penalidade_retorno * erro_retorno * retorno_medio

            gradiente_total = grad_risco + grad_retorno

            pesos_novo = pesos - taxa_aprendizado * gradiente_total

            ret_novo = np.clip(pesos_novo, 0, 1)
            ret_novo = ret_novo / np.sum(ret_novo) if np.sum(ret_novo) > 0 else pesos
            ret_novo_val = ret_novo @ retorno_medio

            if abs(ret_novo_val - r_alvo) < abs(ret_atual - r_alvo):
                pesos = ret_novo



            if abs(erro_retorno) < 0.0001 and iteracao > 1000:
                break

        pesos = np.clip(pesos, 0, 1)
        pesos = pesos / np.sum(pesos)
    else:
        pesos = pesos_lagrange

    retorno = pesos @ retorno_medio
    risco = np.sqrt(pesos @ covariancia @ pesos)
    sharpe = (retorno - taxa_livre_risco) / (risco) if risco > 1e-10 else 0.0

    return pesos, retorno, risco, sharpe

# OBTER SELIC
def obter_selic_atual():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json'
    try:
        response = requests.get(url, timeout=5)
        dados = response.json()
        selic_decimal = float(dados[0]['valor']) / 100
        data = dados[0]['data']
        return selic_decimal
    except Exception as e:
        print(f"Erro: {e}")
        print("   Usando Selic padrão: 10.75%")
        return 0.1075
    
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
    

#===================================================================
#   DASHBOARD INTERATIVO PARA ESCOLHA E RESULTADO DO PORTFÓLIO
#===================================================================
@st.cache_data
def indice_tickers():
    url = "https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBXL?language=pt-br"
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        
        wait = WebDriverWait(driver, 10) 
        
        elemento_dropdown = wait.until(
            EC.visibility_of_element_located((By.ID, "selectPage"))
        )
        
        select = Select(elemento_dropdown)
        select.select_by_visible_text("60")

        time.sleep(5) 

        html_da_pagina = driver.page_source
        
        tabela_ibov = pd.read_html(html_da_pagina)[0][:-2]
        tickers_lista = list(tabela_ibov['Código'])
        nome_acoes = list(tabela_ibov['Ação'])
        return tickers_lista, nome_acoes
        
    except (ValueError, KeyError):
        return [], []
    except Exception as e:
        return [], []
    finally:
        driver.quit()
       
ibv_50, nome_ibv_50 = indice_tickers()
print(ibv_50)
print(nome_ibv_50)

def Info():
    st.set_page_config(
        page_title="Informações",
        page_icon="❓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(" ❓ Informações")
    st.markdown("---")
    
    st.markdown(" #### Empresas listadas")
    tabela_info = pd.DataFrame({'Tickers': ibv_50, "Ações": nome_ibv_50})
    st.dataframe(tabela_info, hide_index = True)
    with st.container():
        st.markdown("""
        ## Otimizadores:
        Este painel utiliza dois métodos de otimização de portfólio baseado na Teoria Moderna do Portfólio (Markowitz). 
        O objetivo principal do primeiro método é construir a carteira de Mínima Variância (menor risco) para um nível de retorno-alvo que você define.
        Enquanto que para o segundo método, o objetivo é ...

        ### Método 1: Uma Abordagem Híbrida
        O algoritmo opera em duas etapas para encontrar a melhor alocação de ativos:

        **1. A Solução Analítica (Lagrange)**
        
        Primeiro, o otimizador tenta encontrar a solução "matematicamente perfeita" usando um método clássico (Multiplicadores de Lagrange). Esta solução calcula a alocação de ativos que minimiza o risco para o seu retorno-alvo em um mundo ideal.

        **2. A Verificação de Realidade e Otimização Numérica**

        * **O Problema:** A solução "perfeita" muitas vezes inclui alocações irreais para um investidor comum (venda a descoberto com pesos negativos ou alavancagem com pesos > 100%).
        * **A Solução:** O código verifica se isso aconteceu. Se a solução matemática for irreal, ela é descartada.
        * **O Plano B (Gradient Descent):** O algoritmo ativa um otimizador numérico. Este método:
            * Começa com uma carteira simples (ex: pesos iguais).
            * Itera milhares de vezes, fazendo pequenos ajustes para minimizar o risco e, ao mesmo tempo, atingir o retorno-alvo.
            * Crucialmente, ele força que todos os pesos se mantenham entre 0% e 100% e que a soma total seja 100%.
        
        ### Método 2: 

        ---
        **O Resultado Final:** Você vê a carteira com o menor risco (volatilidade) possível para o retorno desejado, respeitando as restrições do mundo real (sem vender a descoberto).
        """)
    

def Home():
    st.set_page_config(
        page_title="Dashboard de montagem de portfólio",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Portfólio de investimento otimizado")
    st.markdown("---")

    with st.container(horizontal=True, horizontal_alignment="center"):
        selecionadas = st.pills(
            "Selecione os ativos:",
            options=ibv_50,
            selection_mode="multi"
        )

    if not selecionadas:
        st.info("Por favor, selecione um ou mais ativos acima para carregar os dados.")
    else:
        dados = get_dados(selecionadas)
        info_ativos = get_metricas(dados)
        with st.form(key='meu_formulario'):
            col1,col2 = st.columns(2)
            with col1:
                ret_alvo = st.number_input("Retorno Alvo (entre 0 e 1):", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.3f")
            with col2:
                capital = st.number_input("Qual valor deseja investir? (Ex: 1000.00)", min_value = 0.0)
            submit_button = st.form_submit_button(label='Rodar')
            
        st.markdown("---")

        if submit_button:
            col1, col2= st.columns(2)
            with col1:
                ax = graficoAcoes(dados)
                st.pyplot(ax.get_figure())
            with col2:
                ax2 = graficoVolatilidade(info_ativos)
                st.pyplot(ax2.get_figure())
                
                
            col3,col4 = st.columns(2)
            with col3:
                ax3 = graficoRAcumulado(info_ativos)
                st.pyplot(ax3.get_figure())
            with col4:
                ax4 = graficoRetorno(info_ativos)
                st.pyplot(ax4.get_figure())
            
            with st.container(border = True):
                st.write("Método de minimização do risco:")
                pesos,retorno,risco,sharpe = otimizacao(info_ativos,ret_alvo)
                investimento = pesos*capital
                dict_pesos = {'Ativos': selecionadas, 'Pesos': pesos, 'Investimento (R$)': investimento}
                col1,col2,col3 = st.columns(3)
                col1.metric(label = "Retorno",
                            value = f'{retorno:.3f}',
                            border = True)
                col2.metric(label = 'Risco',
                            value = f'{risco:.3f}',
                            border = True)
                col3.metric(label = "Sharpe ratio",
                            value = f'{sharpe:.3f}',
                            border = True)
                st.dataframe(dict_pesos)
                
pg = st.navigation([Home,Info], position = 'top')
pg.run()