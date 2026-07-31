import streamlit as st
import pandas as pd

CSV_PATH = 'data/ibxl.csv'
df_ibov = pd.read_csv(CSV_PATH, index_col=0)
ibv_50 = list(df_ibov.index)
nome_ibv_50 = list(df_ibov['acao'])

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
    st.markdown("""

### Visão Geral
Este painel interativo aplica métodos de otimização não linear para construir carteiras eficientes de ações da B3 com base na Teoria Moderna do Portfólio.  
O objetivo é encontrar alocações que maximizem retorno ajustado ao risco, respeitando:

- soma dos pesos igual a 100%;
- proibição de venda a descoberto (wi ≥ 0);
- uso de dados históricos reais.

Os ativos disponíveis são obtidos automaticamente a partir do índice IBXL, e os preços são coletados via *yfinance*.

---

### Formulação do Problema
Para n ativos escolhidos:

- **Retornos médios anuais:** µ  
- **Covariância anualizada:** Σ  
- **Taxa livre de risco:** rf  
- **Pesos:** w

O retorno da carteira é:
**R(w) = µᵀ w**

A volatilidade é:
**σ(w) = √(wᵀ Σ w)**

O índice de Sharpe é:
**S(w) = (µᵀ w – rf) / √(wᵀ Σ w)**

O problema final é:

- minimizar –S(w)  
- sujeito a 1ᵀ w = 1 e w ≥ 0  

A região viável é um **simplexo**, onde os pesos são não-negativos e somam 1.

---

## Métodos de Otimização Implementados

---

### 1. Maximização do Índice de Sharpe (Método Híbrido)

O método foi dividido em duas etapas:

#### (a) Solução Analítica (Irrestrita)
Calcula-se o portfólio tangente:

**w\* = Σ⁻¹(µ − rf·1) / (1ᵀ Σ⁻¹ (µ−rf·1))**

Se w\* satisfaz **w ≥ 0** e **1ᵀw = 1**, a solução é aceita.

#### (b) Solução Numérica: Gradiente Ascendente Projetado
Caso contrário:

- direção via ∇S(w);  
- backtracking com Armijo;  
- projeção no simplexo via algoritmo O(n log n);  
- convergência quando |ΔSharpe| < 10⁻⁸.

Este método mostrou-se eficiente e extremamente estável.

---

### 2. Método de Barreiras Logarítmicas com Penalidade Quadrática

As restrições são tratadas com:

- barreira para wi > 0:  
  **φ(w) = − Σ log(wi)**
- penalidade quadrática para 1ᵀw = 1:  
  **(t/2)(1ᵀ w − 1)²**

Cada subproblema é resolvido por **BFGS**, com t aumentando iterativamente (caminho central).

Apesar da fundamentação sólida, o método apresentou:

- instabilidade numérica em dimensões maiores;
- gradiente e Hessiana mal-condicionados perto de wi → 0.

---

### 3. Método do Gradiente Espectral Projetado (SPG)

Combina:

- projeção no simplexo (sempre viável),
- direção espectral de Barzilai–Borwein,
- busca linear com backtracking.

Define:

- sk = wk − wk−1  
- yk = ∇f(wk) − ∇f(wk−1)  
- passo espectral:  
  **λk = (sᵀs)/(sᵀy)**  
- direção:  
  **dk = P(wk − λk∇f(wk)) − wk**

Este método teve o melhor desempenho geral, especialmente em dimensões maiores.

---

### Observações

- SPG foi **o método mais eficiente** e com o melhor compromisso entre precisão e tempo.
- O método de barreiras se degradou rapidamente conforme n aumentou.
- O gradiente projetado é robusto, porém mais lento em n=25.
- SPG converge com cerca de **metade das iterações** do gradiente projetado.

---

## Conclusão

Os experimentos mostraram que:

- Métodos baseados em barreiras têm forte fundamentação teórica, mas sofrem com:
  - instabilidade numérica,
  - sensibilidade a hiperparâmetros,
  - custo computacional elevado.

Por outro lado:

- Métodos baseados em projeção se mostraram **dominantes** para o problema:
  - garantem viabilidade estrita,
  - não exigem parâmetros artificiais,
  - são mais estáveis.

Entre eles, o **Método SPG** apresentou o melhor desempenho:

- passos longos e eficientes,
- comportamento de segunda ordem sem calcular a Hessiana,
- custo computacional baixo,
- excelente estabilidade.

**Conclusão geral:** SPG é a abordagem recomendada para otimização de portfólios com restrições práticas na B3.

## Referências

[1] ASSAF NETO, Alexandre. *Mercado Financeiro*. 9. ed. São Paulo: Atlas, 2008.

[2] CHOEY, Mark; WEIGEND, Andreas S. *Nonlinear Trading Models Through Sharpe Ratio Maximization*.  
Leonard N. Stern School of Business, New York University, 1996.

[3] DUCHI, J.; SHALEV-SHWARTZ, S.; SINGER, Y.; CHANDRA, T.  
*Efficient Projections onto the L1-Ball for Learning in High Dimensions*.  
In: Proceedings of the 25th International Conference on Machine Learning (ICML). ACM, 2008.  
Disponível em: https://dl.acm.org/doi/10.1145/1390156.1390191.  
Acesso em: 02 dez. 2025.

[4] ELTON, Edwin J.; GRUBER, Martin J.; BROWN, Stephen; GOETZMANN, William.  
*Moderna teoria de carteira e análise de investimentos*. São Paulo: Elsevier, 2012.

[5] MARKOWITZ, Harry. *Portfolio Selection*. The Journal of Finance, 1952.

[6] MEURER, Aaron; SMITH, Christopher P.; PAPAGHEORGHIOS, Mateusz; et al.  
*SymPy: symbolic computing in Python*. PeerJ Computer Science, v. 3, n. e103, 2017.

[7] NOCEDAL, Jorge; WRIGHT, Stephen J. *Numerical Optimization*.  
2. ed. New York: Springer, 2006.

[8] QUANTPEDIA. *Markowitz Model*.  
Disponível em: https://quantpedia.com/markowitz-model/#:~:text=Tangency%20portfolio%2C%20the%20red%20point.  
Acesso em: 30 nov. 2025.

[9] THE SCIPY COMMUNITY. *scipy.optimize.minimize*. SciPy v1.11.4 Reference Guide.  
Disponível em: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.  
Acesso em: 02 dez. 2025.

[10] WIKIPEDIA. *Simplex*.  
Disponível em: https://en.wikipedia.org/wiki/Simplex.  
Acesso em: 30 nov. 2025.

"""
)