import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def ordered_corr(df, target):
    # 1. Calcular correlação com label
    corr_with_label = df.corr()[target].drop(target)

    # 2. Ordenar colunas por correlação com label (exceto o próprio label)
    sorted_cols = corr_with_label.abs().sort_values(ascending=False).index.tolist()

    # 3. Reordenar DataFrame: colunas ordenadas + target
    df_sorted = df[sorted_cols + [target]]

    # 4. Recalcular correlações
    corr = df_sorted.corr(method='pearson')
    p_values = df_sorted.corr(method=lambda x, y: stats.pearsonr(x, y)[1])

    return corr, p_values

def plot_corr(corr, p_values, highlight_p=True, p_thresh=0.05):
    # máscara para metade superior
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # monta anotação: negrito quando p < 0.05 (exceto diagonal)
    annot = None
    if highlight_p:
        annot = np.empty_like(corr, dtype=object)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr.iloc[i, j]
                if i != j and p_values.iloc[i, j] < p_thresh:
                    annot[i, j] = rf"$\mathbf{{{val:.2f}}}$"
                else:
                    annot[i, j] = f"{val:.2f}"

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,     # usamos nossas strings
        fmt="",          # não reformatar
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    plt.title('Matriz de Correlação (valores em negrito quando p < 0.05)', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    plt.show()

def remove_highly_correlated(df_corr: pd.DataFrame, target_col: str, threshold: float = 0.5):
    """
    Remove features altamente correlacionadas, começando das mais relevantes para o target.
    
    Args:
        df_corr: DataFrame com a matriz de correlação
        target_col: Nome da coluna alvo
        threshold: Limiar de correlação para remoção (absoluto)
        
    Returns:
        Lista de colunas a serem mantidas
    """
    # Ordenar features por correlação absoluta com o target (decrescente)
    corr_with_target = df_corr[target_col].drop(target_col).abs().sort_values(ascending=False)
    features_sorted = corr_with_target.index.tolist()
    
    to_keep = []
    to_drop = set()
    
    for feature in features_sorted:
        if feature in to_drop:
            continue
            
        to_keep.append(feature)
        
        # Encontrar features altamente correlacionadas com a feature atual
        corr_with_feature = df_corr[feature].drop([*to_keep, *to_drop, target_col])
        highly_correlated = corr_with_feature[abs(corr_with_feature) > threshold].index.tolist()
        
        if highly_correlated:
            print(f"Feature '{feature}' (corr={df_corr.loc[feature, target_col]:.3f}) removeu:")
            for fc in highly_correlated:
                print(f"  - {fc} (corr={df_corr.loc[feature, fc]:.3f} com {feature})")
                to_drop.add(fc)
    
    print(f"\nTotal de features mantidas: {len(to_keep)}")
    print(f"Total de features removidas: {len(to_drop)}")
    return [*to_keep, target_col]