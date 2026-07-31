# Dashboard de Otimização de Portfólio - IBrX

Dashboard interativo desenvolvido em Python e Streamlit para a otimização e alocação de ativos baseados no índice IBrX da B3. O sistema permite a seleção de uma cesta de ações e calcula a alocação ideal de capital para maximizar o retorno e minimizar o risco, aplicando métodos de finanças quantitativas e modelagem estatística.

## Destaque Técnico: Implementação Matemática Customizada

O núcleo de otimização deste projeto foi desenvolvido integralmente sem o uso de solvers comerciais ou bibliotecas prontas de resolução. A rotina de convergência, o cálculo da fronteira eficiente e a aplicação de restrições foram implementados construindo os métodos numéricos e a álgebra linear diretamente no código, garantindo controle matemático e estatístico total sobre o modelo.

## Funcionalidades

* **Seleção Personalizada:** Escolha de ativos específicos listados no índice IBrX da B3.
* **Algoritmos Customizados:** Otimização matemática nativa baseada em métodos numéricos próprios.
* **Restrições de Alocação:** Aplicação de limites para garantir a diversificação do portfólio.
* **Visualização de Dados:** Gráficos interativos de desempenho histórico, matriz de covariância e evolução da carteira.

## Tecnologias Utilizadas

* Python
* Streamlit
* Pandas
* NumPy
* yfinance

## Instalação

```
git clone [https://github.com/josefabiobrito/Projeto-Para-Otimizar-Ativos.git](https://github.com/josefabiobrito/Projeto-Para-Otimizar-Ativos.git)
cd Projeto-Para-Otimizar-Ativos
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Como Executar
```
streamlit run main.py
```

## Estrutura do Projeto

├── analises/

│   ├── analise_numerica.py

│   ├── trajetoria_visualizacao.py

├── app/

│   ├── Home.py

│   ├── Info.py

├── data/

│   ├── ibxl.csv

│   ├── ibxl_meta.json

├── src/

│   ├── main.py

│   ├── packages.txt

│   └── requirements.txt

└── README.md

## Contribuição
Sinta-se à vontade para abrir issues e enviar pull requests. 
Contribuições voltadas para a otimização dos processos estocásticos ou melhorias de performance nos cálculos matriciais são muito bem-vindas.
