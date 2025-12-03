"""
Arquivo com funções de suporte para operações com os dados em .parquet
"""
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import shutil

def _infer_metadata_from_path(path: Union[str, Path]) -> Tuple[str, str, int]:
    """Infere metadados do arquivo a partir do caminho.

    Extrai sensor, banda e label com base na estrutura de pastas do caminho.
    Função interna usada por outras funções do módulo.

    Args:
        path (str or Path): Caminho para o arquivo .parquet.

    Returns:
        Tuple[str, str, int]: Tupla com (sensor, banda, label).
            - sensor: 'current' ou 'voltage' ou 'unknown'.
            - banda: 'HF', 'LF' ou 'NA' (para dados de calibração).
            - label: 1 (HIF) ou -1 (CAL).
    """
    p = Path(path)
    parts = [part for part in p.parts]
    sensor = 'unknown'
    banda = 'unknown'
    label = -1
    if 'Current_HIF' in parts:
        sensor = 'current'
        label = 1
        if 'HF' in parts:
            banda = 'HF'
        elif 'LF' in parts:
            banda = 'LF'
    elif 'Voltage_HIF' in parts:
        sensor = 'voltage'
        label = 1
        if 'HF' in parts:
            banda = 'HF'
        elif 'LF' in parts:
            banda = 'LF'
    elif 'Currents_cal' in parts:
        sensor = 'current'
        # Se existir subpasta HF/LF, respeitar. Caso contrário, NA.
        if 'HF' in parts:
            banda = 'HF'
        elif 'LF' in parts:
            banda = 'LF'
        else:
            banda = 'NA'
        label = -1
    elif 'Voltages_cal' in parts:
        sensor = 'voltage'
        # Se existir subpasta HF/LF, respeitar. Caso contrário, NA.
        if 'HF' in parts:
            banda = 'HF'
        elif 'LF' in parts:
            banda = 'LF'
        else:
            banda = 'NA'
        label = -1
    return sensor, banda, label

def _infer_condicao_from_path(path: Union[str, Path]) -> str:
    """Infere a condição (HIF ou CAL) a partir do caminho do arquivo.

    Args:
        path (str or Path): Caminho para o arquivo.

    Returns:
        str: 'HIF' para falha, 'CAL' para calibração, ou 'unknown'.
    """
    p = Path(path)
    parts = [part for part in p.parts]
    if 'Current_HIF' in parts or 'Voltage_HIF' in parts:
        return 'HIF'
    if 'Currents_cal' in parts or 'Voltages_cal' in parts:
        return 'CAL'
    return 'unknown'

def listar_arquivos_parquet(data_dir: Union[str, Path]) -> dict:
    """Lista arquivos .parquet agrupados por chave de metadados.

    Varre recursivamente os subdiretórios em busca de arquivos .parquet e os agrupa
    por uma chave no formato 'sensor/condicao/banda/label'.

    Args:
        data_dir (str or Path): Diretório contendo os dados.

    Returns:
        dict: Dicionário onde as chaves são strings no formato 'sensor/condicao/banda/label'
              e os valores são listas de caminhos para os arquivos correspondentes.

    Examples:
        >>> arquivos = listar_arquivos_parquet('caminho/para/projeto')
        >>> for chave, paths in arquivos.items():
        ...     print(f"{chave}: {len(paths)} arquivos")
    """
    base = Path(data_dir)
    padroes = [
        base / 'Current_HIF' / 'HF' / '*.parquet',
        base / 'Current_HIF' / 'LF' / '*.parquet',
        base / 'Currents_cal' / '*.parquet',
        base / 'Voltage_HIF' / 'HF' / '*.parquet',
        base / 'Voltage_HIF' / 'LF' / '*.parquet',
        base / 'Voltages_cal' / 'HF' / '*.parquet',
        base / 'Voltages_cal' / 'LF' / '*.parquet',
    ]
    arquivos = []
    for padrao in padroes:
        arquivos.extend(sorted([str(p) for p in padrao.parent.glob(padrao.name)]))
    out = {}
    for f in arquivos:
        sensor, banda, label = _infer_metadata_from_path(f)
        cond = _infer_condicao_from_path(f)
        chave = f"{sensor}/{cond}/{banda}/{label}"
        out.setdefault(chave, []).append(f)
    return out

def carregar_parquet(caminho: Union[str, Path], colunas: Optional[List[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Carrega um arquivo parquet para um DataFrame pandas.

    Args:
        caminho (str or Path): Caminho para o arquivo .parquet.
        colunas (list of str, optional): Lista de colunas para carregar. Se None, carrega todas.
        nrows (int, optional): Número máximo de linhas a serem carregadas.

    Returns:
        pd.DataFrame: DataFrame com os dados do arquivo.

    Examples:
        >>> df = carregar_parquet('dados.parquet', colunas=['col1', 'col2'], nrows=1000)
    """
    df = pd.read_parquet(caminho, columns=colunas)
    if nrows is not None:
        df = df.head(nrows)
    return df

def preview_parquet(caminho: Union[str, Path], n: int = 5) -> dict:
    """Fornece uma prévia rápida dos dados em um arquivo parquet.

    Args:
        caminho (str or Path): Caminho para o arquivo .parquet.
        n (int, optional): Número de linhas para incluir no head. Padrão: 5.

    Returns:
        dict: Dicionário com:
            - 'shape': Tupla (linhas, colunas)
            - 'dtypes': Mapeamento de nomes de colunas para tipos de dados
            - 'missing': Contagem de valores ausentes por coluna
            - 'head': Primeiras n linhas do DataFrame

    Examples:
        >>> info = preview_parquet('dados.parquet', n=3)
        >>> print(f"Shape: {info['shape']}")
        >>> print(info['dtypes'])
    """
    df = carregar_parquet(caminho)
    info = {
        'shape': df.shape,
        'dtypes': {k: str(v) for k, v in df.dtypes.items()},
        'missing': df.isna().sum().to_dict(),
        'head': df.head(n),
    }
    return info

def descrever_dataframe(df: pd.DataFrame) -> dict:
    """Gera estatísticas descritivas para um DataFrame.

    Calcula estatísticas separadamente para colunas numéricas e categóricas,
    além de fornecer informações sobre valores ausentes e correlações.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        dict: Dicionário com:
            - 'describe_numerico': Estatísticas descritivas para colunas numéricas
            - 'describe_categorico': Estatísticas para colunas categóricas
            - 'missing': Contagem de valores ausentes por coluna
            - 'correlacao': Matriz de correlação entre colunas numéricas

    Examples:
        >>> df = pd.DataFrame({'num': [1, 2, 3], 'cat': ['a', 'b', 'a']})
        >>> stats = descrever_dataframe(df)
        >>> print(stats['describe_numerico'])
    """
    numericas = df.select_dtypes(include=['number'])
    categoricas = df.select_dtypes(exclude=['number'])
    resumo = {
        'describe_numerico': numericas.describe().T if not numericas.empty else pd.DataFrame(),
        'describe_categorico': categoricas.describe().T if not categoricas.empty else pd.DataFrame(),
        'missing': df.isna().sum().to_dict(),
        'correlacao': numericas.corr(numeric_only=True) if not numericas.empty else pd.DataFrame(),
    }
    return resumo

def carregar_dataset(
    base_dir: Union[str, Path],
    limite_por_grupo: Optional[int] = None,
    incluir_label: bool = True,
    adicionar_origem: bool = True,
) -> pd.DataFrame:
    """Carrega e concatena múltiplos arquivos parquet em um único DataFrame.

    ATENÇÃO: Pode consumir muita memória para conjuntos de dados grandes.
    Considere usar iterar_dataset() para processamento por lote.

    Args:
        base_dir (str or Path): Diretório raiz contendo os dados.
        limite_por_grupo (int, optional): Número máximo de arquivos por grupo.
        incluir_label (bool, optional): Se True, adiciona coluna 'label'.
        adicionar_origem (bool, optional): Se True, adiciona colunas de metadados.

    Returns:
        pd.DataFrame: DataFrame concatenado com os dados de todos os arquivos.

    Examples:
        # Carregar apenas 2 arquivos de cada grupo
        >>> df = carregar_dataset('caminho/para/dados', limite_por_grupo=2)
    """
    grupos = listar_arquivos_parquet(base_dir)
    amostras = []
    for chave, arquivos in grupos.items():
        count = 0
        for arq in arquivos:
            if limite_por_grupo is not None and count >= limite_por_grupo:
                break
            df = carregar_parquet(arq)
            sensor, banda, label = _infer_metadata_from_path(arq)
            cond = _infer_condicao_from_path(arq)
            if incluir_label:
                df = df.copy()
                df['label'] = label
            if adicionar_origem:
                df = df.copy()
                df['sensor'] = sensor
                df['banda'] = banda
                df['condicao'] = cond
                df['arquivo'] = str(arq)
            amostras.append(df)
            count += 1
    if not amostras:
        return pd.DataFrame()
    return pd.concat(amostras, ignore_index=True)

def iterar_arquivos_parquet(base_dir: Union[str, Path], limite_por_grupo: Optional[int] = None):
    """Itera sobre arquivos parquet com metadados, sem carregar os dados.

    Útil para inspecionar a estrutura dos dados antes de carregá-los.

    Args:
        base_dir (str or Path): Diretório raiz contendo os dados.
        limite_por_grupo (int, optional): Número máximo de arquivos por grupo.

    Yields:
        tuple: (caminho_arquivo, metadados), onde metadados é um dicionário com:
            - 'arquivo': caminho completo do arquivo
            - 'nome': nome do arquivo (sem diretório)
            - 'sensor': 'current' ou 'voltage'
            - 'banda': 'HF', 'LF' ou 'NA'
            - 'condicao': 'HIF' ou 'CAL'
            - 'label': 1 (HIF) ou -1 (CAL)

    Examples:
        >>> for caminho, meta in iterar_arquivos_parquet('dados'):
        ...     print(f"{meta['condicao']}: {meta['arquivo']}")
    """
    grupos = listar_arquivos_parquet(base_dir)
    for _, arquivos in grupos.items():
        count = 0
        for arq in arquivos:
            if limite_por_grupo is not None and count >= limite_por_grupo:
                break
            file_name = os.path.basename(arq)
            sensor, banda, label = _infer_metadata_from_path(arq)
            cond = _infer_condicao_from_path(arq)
            meta = {
                'arquivo': str(arq),
                'nome': file_name,
                'sensor': sensor,
                'banda': banda,
                'condicao': cond,
                'label': label,
            }
            yield arq, meta
            count += 1

def iterar_dataset(
    base_dir: Union[str, Path],
    limite_por_grupo: Optional[int] = None,
    incluir_label: bool = True,
    adicionar_origem: bool = True,
    nrows: Optional[int] = None,
):
    """Itera sobre DataFrames carregados de arquivos parquet.

    Carrega um arquivo por vez, processa e descarta da memória, ideal para grandes conjuntos.

    Args:
        base_dir (str or Path): Diretório raiz contendo os dados.
        limite_por_grupo (int, optional): Número máximo de arquivos por grupo.
        incluir_label (bool, optional): Se True, adiciona coluna 'label'.
        adicionar_origem (bool, optional): Se True, adiciona colunas de metadados.
        nrows (int, optional): Número máximo de linhas a carregar por arquivo.

    Yields:
        tuple: (df, metadados) para cada arquivo processado.

    Examples:
        >>> for df, meta in iterar_dataset('dados', nrows=1000):
        ...     print(f"Processando {meta['arquivo']} com {len(df)} linhas")
        ...     # Extrair features e salvar resultados
    """
    for caminho, meta in iterar_arquivos_parquet(base_dir, limite_por_grupo=limite_por_grupo):
        df = carregar_parquet(caminho, nrows=nrows)
        if incluir_label:
            df = df.copy()
            df['label'] = meta['label']
        if adicionar_origem:
            df = df.copy()
            df['sensor'] = meta['sensor']
            df['banda'] = meta['banda']
            df['condicao'] = meta['condicao']
            df['arquivo'] = meta['arquivo']
        yield df, meta

def contar_classes(df: pd.DataFrame, coluna: str = 'label') -> dict:
    """Conta a ocorrência de cada classe em uma coluna do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        coluna (str, optional): Nome da coluna com as classes. Padrão: 'label'.

    Returns:
        dict: Dicionário com as contagens de cada valor único na coluna.
              Retorna {} se a coluna não existir.

    Examples:
        >>> df = pd.DataFrame({'label': [1, 1, -1, 1, -1]})
        >>> contar_classes(df)
        {1: 3, -1: 2}
    """
    if coluna not in df.columns:
        return {}
    return df[coluna].value_counts().to_dict()

def plotar_sinal(df: pd.DataFrame, colunas: Optional[List[str]] = None, n: int = 1000, ax=None):
    """Plota um sinal temporal a partir de um DataFrame.

    Se nenhum eixo for fornecido, cria uma nova figura. Se não forem especificadas
    colunas, plota as 3 primeiras colunas numéricas encontradas.

    Args:
        df (pd.DataFrame): DataFrame com os dados a serem plotados.
        colunas (list of str, optional): Nomes das colunas para plotar.
        n (int, optional): Número máximo de pontos a plotar. Padrão: 1000.
        ax (matplotlib.axes.Axes, optional): Eixo para desenhar o gráfico.

    Returns:
        matplotlib.figure.Figure or matplotlib.axes.Axes:
            Retorna a figura se ax=None, senão retorna o eixo modificado.

    Examples:
        >>> import numpy as np
        >>> df = pd.DataFrame({'s1': np.random.randn(100), 's2': np.random.randn(100)})
        >>> fig = plotar_sinal(df, colunas=['s1', 's2'], n=50)
        >>> plt.show()
    """
    if df.empty:
        return ax
    if colunas is None:
        colunas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        colunas = colunas[:3]
    dados = df[colunas].head(n)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        dados.plot(ax=ax)
        ax.set_title('Sinal (prévia)')
        ax.legend(loc='best')
        return fig
    else:
        dados.plot(ax=ax)
        ax.set_title('Sinal (prévia)')
        ax.legend(loc='best')
        return ax

def organizar_arquivos_por_banda(diretorio_origem):
    """
    Organiza arquivos .parquet em pastas HF/LF e remove os sufixos dos nomes.
    
    Args:
        diretorio_origem (str): Caminho para o diretório contendo os arquivos .parquet
        
    Exemplo:
        >>> organizar_arquivos_por_banda('Dados/raw/Voltages_cal')
    """
    # Converter para Path e garantir que é um diretório
    dir_origem = Path(diretorio_origem)
    if not dir_origem.is_dir():
        raise ValueError(f"O diretório {dir_origem} não existe")
    
    # Criar pastas de destino se não existirem
    dir_hf = dir_origem.parent / 'HF'
    dir_lf = dir_origem.parent / 'LF'
    dir_hf.mkdir(exist_ok=True)
    dir_lf.mkdir(exist_ok=True)
    
    # Contadores para relatório
    movidos_hf = 0
    movidos_lf = 0
    ignorados = 0
    
    # Processar cada arquivo .parquet
    for arquivo in dir_origem.glob('*.parquet'):
        nome = arquivo.stem  # Nome sem extensão
        sufixo = nome.split('_')[-1]  # Pega HF ou LF do final
        
        # Determinar diretório de destino
        if sufixo.upper() == 'HF':
            dir_destino = dir_hf
            movidos_hf += 1
        elif sufixo.upper() == 'LF':
            dir_destino = dir_lf
            movidos_lf += 1
        else:
            print(f"Arquivo ignorado (sem sufixo HF/LF): {arquivo.name}")
            ignorados += 1
            continue
        
        # Novo nome sem o sufixo _HF ou _LF
        novo_nome = '_'.join(nome.split('_')[:-1]) + '.parquet'
        caminho_destino = dir_destino / novo_nome
        
        # Mover o arquivo
        try:
            shutil.move(str(arquivo), str(caminho_destino))
            print(f"Movido: {arquivo.name} -> {caminho_destino}")
        except Exception as e:
            print(f"Erro ao mover {arquivo.name}: {e}")
    
    # Resumo
    print("\nResumo da operação:")
    print(f"- Arquivos movidos para HF: {movidos_hf}")
    print(f"- Arquivos movidos para LF: {movidos_lf}")
    print(f"- Arquivos ignorados: {ignorados}")