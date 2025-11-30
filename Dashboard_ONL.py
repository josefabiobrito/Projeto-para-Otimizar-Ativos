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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
import platform
pd.set_option("display.min_rows",50)

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    if platform.system() == "Linux":
        service = Service(executable_path='/usr/bin/chromedriver')
    else:
        service = Service()
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

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
def otimizacao_sharpe_manual(metricas):
   
    taxa_livre_risco = obter_selic_atual()
    covariancia = metricas['covariancia']
    retorno_medio = metricas['retorno_medio']
    n_ativos = len(retorno_medio)
    
    try:
        cov_inv = np.linalg.inv(covariancia)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(covariancia)
        
    excesso_retorno = retorno_medio - taxa_livre_risco
    pesos_analiticos = cov_inv @ excesso_retorno
    
    if np.sum(pesos_analiticos) != 0:
        pesos_analiticos /= np.sum(pesos_analiticos) 
    
    if np.all(pesos_analiticos >= -1e-6) and np.all(pesos_analiticos <= 1 + 1e-6):
        retorno = pesos_analiticos @ retorno_medio
        risco = np.sqrt(pesos_analiticos @ covariancia @ pesos_analiticos)
        sharpe = (retorno - taxa_livre_risco) / risco
        return pesos_analiticos, retorno, risco, sharpe

    pesos = np.ones(n_ativos) / n_ativos
    
    taxa_aprendizado = 0.35  
    max_iter = 5000
    tolerancia = 1e-6
    sharpe_anterior = -np.inf

    for i in range(max_iter):
        ret_port = pesos @ retorno_medio
        var_port = pesos @ covariancia @ pesos
        vol_port = np.sqrt(var_port)
        
        if vol_port < tolerancia: break
            
        sharpe_atual = (ret_port - taxa_livre_risco) / vol_port
        
        delta_sharpe = sharpe_atual - sharpe_anterior
        
        if abs(delta_sharpe) < tolerancia and i > 100:
            break
        sharpe_anterior = sharpe_atual
        
        grad_numerador = retorno_medio
        grad_denominador = (covariancia @ pesos) / vol_port 
        
        gradiente = (grad_numerador * vol_port - (ret_port - taxa_livre_risco) * grad_denominador) / (var_port)

        pesos_novos = pesos + taxa_aprendizado * gradiente

        pesos_novos = np.clip(pesos_novos, 0.01, 0.99)
        
        soma = np.sum(pesos_novos)
        if soma > 0:
            pesos_novos /= soma
        else:
            pesos_novos = np.ones(n_ativos) / n_ativos
            
        pesos = pesos_novos
        
        if i % 50 == 0:
            taxa_aprendizado *= 0.95

    retorno = pesos @ retorno_medio
    risco = np.sqrt(pesos @ covariancia @ pesos)
    sharpe = (retorno - taxa_livre_risco) / risco

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
    driver = get_driver()
    
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
        O objetivo principal do primeiro método é construir a carteira de Mínima Variância (menor risco) através da maximização do índice de Sharpe.

        ### Método 1:
        #### A Solução Analítica (Portfólio de Tangência)
        
        Primeiro, o otimizador tenta encontrar a solução "matematicamente fechada" através de álgebra linear. 
        Ele calcula o ponto exato de tangência da fronteira eficiente (fórmula de Markowitz inversa), 
        buscando o máximo Sharpe teórico instantaneamente, sem necessidade de iterações.

        #### A Verificação de Restrições e Otimização Numérica

        * O Problema: A solução matemática direta frequentemente viola as regras de um fundo comum (sugerindo pesos negativos/venda a descoberto ou alavancagem acima de 100%).
        * A Validação: O código verifica se os pesos analíticos são viáveis. Se violarem as regras, essa solução é descartada.
        * O Plano B (Projected Gradient Ascent): O algoritmo ativa um otimizador numérico de Subida de Gradiente Projetado. Este método:
            * Inicializa com uma carteira equiponderada e calcula a derivada do Índice de Sharpe para determinar a direção de subida.
            * Aplica um Decaimento de Taxa de Aprendizado (*Decay*), reduzindo o tamanho dos passos ao longo do tempo para garantir um ajuste fino e evitar oscilações no topo.
            * Executa uma Projeção no Simplex a cada passo, forçando matematicamente que os pesos negativos sejam zerados e que a soma total retorne a 100%.
            * Define a convergência pela estabilização da Função Objetivo (Sharpe) e não apenas dos pesos, garantindo robustez no resultado final.
        ### Método 2: 

        ---
        ## **O Resultado Final**: 
        Ao final, o otimizador retorna os pesos finais dos ativos, o retorno esperado, o risco (desvio padrão) e o índice de Sharpe da carteira otimizada. 
        Esses valores são exibidos no painel, permitindo ao usuário visualizar a alocação ideal para seu portfólio com base nos ativos selecionados.
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
            capital = st.number_input("Qual valor deseja investir? (Ex: 1000.00)", min_value = 0.0, step = 100.00)
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
                pesos,retorno,risco,sharpe = otimizacao_sharpe_manual(info_ativos)
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