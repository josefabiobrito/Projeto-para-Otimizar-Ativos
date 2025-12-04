import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.Dados import get_dados, get_metricas, obter_ibxl, obter_selic_atual
from src.Visualizacoes import graficoAcoes, graficoRetorno, graficoVolatilidade, graficoRAcumulado, plot_correlacao
from src.BarreiraLogaritmicaPenalizacaoQuadratica import BarreiraLogaritmicaPenalizacaoQuadratica
from src.Analitico import Analitico
from src.GradienteAscendenteProjetado import GradienteAscendenteProjetado
from src.GradienteEspectralProjetado import GradienteEspectralProjetado



CSV_PATH = 'data/ibxl.csv'
META_PATH = 'data/ibxl_meta.json'

def pipeline_dados(ibv_path, meta_path):
    obter_ibxl()
    df_ibov = pd.read_csv(ibv_path, index_col=0)
    ibv_50 = list(df_ibov.index)
    return ibv_50

ibv_50 = pipeline_dados(CSV_PATH, META_PATH)

metodos = {
    "Barreira Logaritmica Penalizacao Quadratica": BarreiraLogaritmicaPenalizacaoQuadratica,
    "Gradiente Espectral Projetado": GradienteEspectralProjetado,
    "Gradiente Ascendente": GradienteAscendenteProjetado,
    "Analitico": Analitico
}

if "r_f" not in st.session_state:
    st.session_state["r_f"] = 0.05  # valor padrão

# Função que obtém o valor da taxa livre de risco
def atualizar_taxa_risco():
    nova_rf = obter_selic_atual()  # sua função externa
    st.session_state["r_f"] = nova_rf

def Home():
    st.set_page_config(
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Portfólio de Investimento Otimizado")
    st.markdown("---")

    with st.container(horizontal=True, horizontal_alignment="center"):
        selecionadas = st.pills(
            "Selecione os ativos:",
            options=ibv_50,
            selection_mode="multi",
            default=["WEGE3", "MULT3", "SUZB3"] 
        )

        if not selecionadas:
            st.info("Por favor, selecione um ou mais ativos acima para carregar os dados.")
        else:
            # Inicializa r_f no session_state
            if "r_f" not in st.session_state:
                st.session_state["r_f"] = 0.05

            # Função para atualizar SELIC
            def atualizar_taxa_risco_callback():
                nova_rf = obter_selic_atual()
                st.session_state["r_f"] = nova_rf

            dados = get_dados(selecionadas)
            info_ativos = get_metricas(dados)

            # ÚNICO FORMULÁRIO
            with st.form(key='meu_formulario'):
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    r_f = st.number_input(
                        "Taxa livre de Risco",
                        key="r_f",
                        step=0.01,
                        format="%.4f"
                    )

                with col2:
                    capital = st.number_input(
                        "Qual valor deseja investir? (Ex: 1000.00)",
                        min_value=0.0,
                        value=1000.00
                    )

                with col3:
                    metodo_escolhido = st.selectbox(
                        "Método de otimização:",
                        options=list(metodos.keys())
                    )

                # Botão Atualizar SELIC dentro do mesmo formulário
                with col5:
                    st.write("Atualizar SELIC")
                    st.form_submit_button(
                        "Atualizar",
                        on_click=atualizar_taxa_risco_callback
                    )

                # Botão de executar
                submit_button = st.form_submit_button("Rodar")

                
    st.markdown("---")

    if submit_button:
        
        with st.container(border = True):
            metricas = get_metricas(dados)
            # r_f = obter_selic_atual()
            cov = metricas['covariancia'].values
            mu = metricas['retorno_medio'].values

            mu = mu.flatten()
            cov = cov

            otimizador = metodos[metodo_escolhido](mu, cov, r_f)

            solucao = otimizador.fit()
            if solucao['success'] == True or not metodo_escolhido == 'Analitico':
                pesos = solucao['w']
                retorno = mu @ pesos 
                risco = np.sqrt(pesos.T @ cov @ pesos)
                sharpe = solucao['sharpe']

                st.markdown(f"### {metodo_escolhido} - {solucao['method']}:")

                investimento = pesos*capital
                dict_pesos = {'Ativos': selecionadas, 'Pesos': pesos, 'Investimento (R$)': investimento}
                col1,col2 = st.columns(2)
                col2.metric(label = "Ganho",
                            value = f'{retorno*capital:.2f} R$',
                            border = True)
                col1.metric(label = 'Montante final',
                            value = f'{capital*(1+retorno):.2f} R$',
                            border = True)
            

                col1,col2,col3 = st.columns(3)
                col1.metric(label = "Retorno Anualizado",
                            value = f'{retorno:.3f}',
                            border = True)
                col2.metric(label = 'Risco',
                            value = f'{risco:.3f}',
                            border = True)
                col3.metric(label = "Índice Sharpe Anualizado",
                            value = f'{sharpe:.3f}',
                            border = True)
                st.dataframe(dict_pesos)
                st.write("Distribuição de pesos:")
                st.bar_chart(data=dict_pesos, x='Ativos', y='Pesos')

                if mu.size == 3:
                    with st.container(border=True):
                        st.markdown("### Otimização de portfólio de investimento com base nos ativos" + f" {selecionadas[0]}, {selecionadas[1]}, {selecionadas[2]}:")
                        col1, col2, col3 = st.columns([1,2,1])  # coluna central maior
                        with col2:

                            def sharpe_ratio(mu, cov, r_f):
                                def f(w):
                                    Rw = w @ mu
                                    var = w @ cov @ w
                                    return (Rw - r_f) / np.sqrt(var)
                                return f

                            f = sharpe_ratio(mu, cov, r_f)

                            N = 200  # menos pontos para acelerar
                            w1 = np.linspace(0, 1, N)
                            w2 = np.linspace(0, 1, N)
                            W1, W2 = np.meshgrid(w1, w2)
                            W3 = 1 - W1 - W2

                            mask = (W1 >= 0) & (W2 >= 0) & (W3 >= 0)
                            Sharpe = np.full_like(W1, np.nan)

                            for i in range(N):
                                for j in range(N):
                                    if mask[i, j]:
                                        Sharpe[i, j] = f([W1[i, j], W2[i, j], W3[i, j]])

                            # figura menor
                            fig, ax = plt.subplots(figsize=(3, 3))
                            c = ax.contourf(W1, W2, Sharpe, levels=20, cmap='plasma')
                            fig.colorbar(c, ax=ax, shrink=0.8)

                            contours = ax.contour(W1, W2, Sharpe, levels=10, colors='black', alpha=0.6, linewidths=0.5)
                            ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

                            ax.set_xlabel(selecionadas[0], fontsize=10)
                            ax.set_ylabel(selecionadas[1], fontsize=10)
                            ax.set_title("Índice Sharpe", fontsize=10)
                            ax.plot([0, 1], [1, 0], color='black', linewidth=0.7, alpha=0.7)
                            ax.set_xticks([0,1])
                            ax.set_yticks([1])
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)


                            # percorre pares consecutivos da trajetória
                            for index in range(1, len(solucao['history'])):
                                p1 = solucao['history'][index-1]
                                p2 = solucao['history'][index]

                                # alpha decresce conforme o passo é mais antigo
                                alpha = (1 - (index / len(solucao['history'])))*0.5

                                # linha tracejada entre p1 e p2
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                        linestyle="--", color="black", alpha=alpha, marker='*', linewidth=0.7, markersize=3)
                                plt.tight_layout()               

                            st.pyplot(fig, use_container_width=False)
                

                    with st.container(border=True):
                        col1, col2 = st.columns(2)
                        tempo = solucao['time']
                        nit = solucao['nit_total']

                        col2.metric(label="Tempo de execução",
                                    value=f"{tempo:.2f} seg")   # ou R$, se for dinheiro

                        col1.metric(label="Número de iterações",
                                    value=f"{nit:.0f}")


                with st.container(border=True):
                    st.markdown("""
                    ### Interpretação das Métricas

                    #### **Retorno anualizado**
                    -  **Baixo**: menor que **5%**
                    -  **Médio**: entre **5% e 10%**
                    -  **Alto**: acima de **10%**

                    #### **Risco (volatilidade anualizada)**
                    -  **Baixo**: menor que **10%**
                    -  **Médio**: entre **10% e 20%**
                    -  **Alto**: acima de **20%**

                    #### **Índice de Sharpe**
                    -  **Ruim**: menor que **1**
                    -  **Aceitável**: entre **1 e 2**
                    -  **Excelente**: acima de **2**
                    """)
            else:
                st.warning("Nenhuma solução encontrada.")