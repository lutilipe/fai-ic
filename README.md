A ordem de execução segue a numeração dos notebooks:

- `1-view_data.ipynb`: Visualização dos dados usando `dados.py` e organização em pastas padronizadas no caso dos dados de Voltagem (CAL)
- `2-reamostragem.ipynb`: Reamostragem dos dados usando `reamostrar.py` com base nas informações de `Dados/dados.md`
- `3-feature_extraction.ipynb`: Extração de features usando `feature_extraction.py` e `correlacao.py` com base nas informações de `features.md`, extraídas de 3 dos artigos de referência da apresentação.
- `4-modelo.ipynb`: Treinamento e avaliação do modelo usando a biblioteca PyCaret.

Com relação aos dados:
- `Dados/dataset_raw.csv`: Saída da reamostragem, mas contem colunas inúteis (nome do arquivo original, frequencia HF ou LF etc)
- `Dados/dataset.csv`: Saída da reamostragem, mas sem as colunas inúteis de `dataset_raw.csv`
- `Dados/dataset_selected.csv`: Saída do processo de seleção de features baseada na correlação de Pearson, feito no arquivo `3-feature_extraction.ipynb`