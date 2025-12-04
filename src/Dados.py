import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from io import StringIO
import numpy as np
import yfinance as yf
import logging

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,  # pode ser DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


DIRETORIO = "data"
CSV_PATH = os.path.join(DIRETORIO, "ibxl.csv")
META_PATH = os.path.join(DIRETORIO, "ibxl_meta.json")

MAX_IDADE_HORAS = 24 * 20   


def precisa_atualizar():
    if not os.path.exists(META_PATH):
        return True
    
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        
        data_str = meta.get("data_extracao")
        data_extracao = datetime.fromisoformat(data_str)        
        idade = datetime.now() - data_extracao
        return idade > timedelta(hours=MAX_IDADE_HORAS)
    
    except:
        return True

def extrair_ibxl_via_selenium():
    url = "https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBXL?language=pt-br"
    inicio = time.time()

    # Selenium headless
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=options)

    try:
        driver.get(url)

        wait = WebDriverWait(driver, 10)

        dropdown = wait.until(EC.visibility_of_element_located((By.ID, "selectPage")))
        select = Select(dropdown)
        select.select_by_visible_text("60")

        time.sleep(4)

        html_page = driver.page_source

        tabela = pd.read_html(StringIO(html_page))[0][:-2]
        mapa_colunas = {
            "Código": "codigo",
            "Ação": "acao",
            "Tipo": "tipo",
            "Qtde. Teórica": "qtde_teorica",
            "Part. (%)": "part_teorica",
            "Part. (%) IBr-X 50": "part_iberf",
        }

        tabela = tabela.rename(columns={orig: novo for orig, novo in mapa_colunas.items() if orig in tabela.columns})

        # tickers = tabela["codigo"].tolist()
        # nomes = tabela["acao"].tolist()

        tempo_exec = round(time.time() - inicio, 2)

        # Criar diretório
        os.makedirs(DIRETORIO, exist_ok=True)

        # Salvar CSV
        tabela.to_csv(CSV_PATH, index=False, encoding="utf-8")

        # Salvar JSON de metadados
        meta = {
            "data_extracao": datetime.now().isoformat(),
            "qtde_registros": len(tabela),
            "tempo_execucao_s": tempo_exec,
            "fonte": url,
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=4)

        return tabela

    finally:
        driver.quit()

def obter_ibxl():
    if not os.path.exists(CSV_PATH) or precisa_atualizar():
        logging.info("Executando Selenium... dados desatualizados")
        tabela = extrair_ibxl_via_selenium()
    else:
        logging.info("Carregando do cache (CSV + JSON)")
        tabela = pd.read_csv(CSV_PATH)

    return tabela

def obter_selic_atual():
    url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json'
    try:
        response = requests.get(url, timeout=5)
        dados = response.json()
        selic_decimal = float(dados[0]['valor']) / 100
        data = dados[0]['data']
        return selic_decimal
    except Exception as e:
        logging.error(f"Erro: {e}")
        logging.error("Usando Selic padrão: 10.75%")
        return 0.1075

def get_dados(tickers, periodo = '4y'):
  tickers_b3 = [t+'.SA' if not t.endswith('.SA') else t for t in tickers]
  dados = yf.download(tickers_b3, auto_adjust=False, period=periodo, interval='1d')
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

if __name__ == '__main__':
    obter_ibxl()
    acoes = pd.read_csv(CSV_PATH)['codigo']
    dados = get_dados(acoes)
    metricas = pd.DataFrame(get_metricas(dados), columns=tickers_b3)
    metricas.to_csv(DIRETORIO+'/ibxl_metricas.csv', index=False, encoding="utf-8")