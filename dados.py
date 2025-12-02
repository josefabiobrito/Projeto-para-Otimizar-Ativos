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

