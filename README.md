# Data-Science
Desafio de ciência de dados para prever o abandono de clientes (churn) de uma instituição financeira.

O dataset contido em *Abandono_clientes.csv* possui 10000 pontos com 12 colunas de features e 1 coluna (Exited) sendo a classificação entre abandono (1) ou não (0).

O objetivo é construir um pipeline de machine learning para prever os 1000 pontos presentes no arquivo *Abandono_teste.csv*.

### Visualizando os dados
Utilize o arquivo *visualizing_data.ipynb* para rodar os algoritmos que ilustram os dados presentes.

### Testando modelos
Alguns modelos de aprendizado de máquina foram testados, como Random Forest, SVM, DNN, etc. No script *testing_models.ipynb* é possível ver quais modelos foram testados e suas acurácias com os dados de validação.

### O pipeline
Depois de visualizar os dados e testar diferentes modelos, foi construído o pipeline que está no arquivo *ML_pipeline.py*. Os resultados das predições dos dados de teste estão salvos em *optimized_predictions.csv*.
