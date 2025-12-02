import streamlit as st
import pandas as pd
from dados import get_dados, get_metricas, graficoAcoes, graficoVolatilidade, graficoRAcumulado, graficoRetorno, otimizacao_sharpe_manual, ibv_50, nome_ibv_50

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
