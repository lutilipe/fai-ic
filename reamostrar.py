import os
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
import math

try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from dados import listar_arquivos_parquet, carregar_parquet, iterar_arquivos_parquet

def inferir_banda_de_nome_ou_caminho(path: str) -> str:
    """Inferir banda ('HF'/'LF') a partir do nome (N_HF.parquet) ou de pastas HF/LF."""
    p = Path(path)
    name = p.stem.upper()
    if name.endswith('_HF'):
        return 'HF'
    if name.endswith('_LF'):
        return 'LF'
    # tentar via pastas
    parts = [s.upper() for s in p.parts]
    if 'HF' in parts:
        return 'HF'
    if 'LF' in parts:
        return 'LF'
    return 'unknown'

def reamostrar_coluna(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Reamostra 1D com anti-alias. Usa resample_poly quando disponível."""
    if fs_out <= 0 or fs_in <= 0:
        raise ValueError("fs_in e fs_out devem ser positivos.")
    
    # Converter para inteiros mantendo a proporção
    if not HAS_SCIPY:
        # fallback simples: média por blocos (não ideal, mas funcional)
        fator = int(round(fs_in / fs_out))
        if fator < 1:
            raise ValueError("Fallback sem scipy não suporta upsampling. Instale scipy.")
        n = len(x) // fator * fator
        return x[:n].reshape(-1, fator).mean(axis=1)
    
    # Garantir que fs_in e fs_out sejam inteiros
    fs_in_int = int(round(fs_in))
    fs_out_int = int(round(fs_out))
    
    # Simplificar a fração
    gcd = np.gcd(fs_out_int, fs_in_int)
    up = fs_out_int // gcd
    down = fs_in_int // gcd
    
    return resample_poly(x, up, down, window=('kaiser', 5.0))

def reamostrar_dataframe(df: pd.DataFrame, fs_in: float, fs_out: float, cols=None) -> pd.DataFrame:
    """Reamostra todas as colunas numéricas (ou subset cols)."""
    if cols is None:
        cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    out = {}
    for c in cols:
        out[c] = reamostrar_coluna(df[c].to_numpy(), fs_in, fs_out)
    # alinhar comprimentos
    lens = {len(v) for v in out.values()}
    L = min(lens) if lens else 0
    for k in out:
        out[k] = out[k][:L]
    return pd.DataFrame(out)

# --- Função para inferir fs_out segundo Nyquist / decimação / fallback ---
def inferir_fs_out_entry(entry_cfg: dict, fs_in: float) -> int:
    """
    entry_cfg pode ser:
      {'fs_in': 10000, 'fs_out': 2000}
    ou
      {'fs_in': 10000, 'f_max': 800}  # quer preservar até 800 Hz -> fs_out = ceil(2*800)=1600
    ou
      {'fs_in': 10000, 'decimation': 5}  # fs_out = fs_in / decimation
    ou apenas {'fs_in': ...} (usar fallback fs_in//2)
    Retorna fs_out (int).
    """
    if 'fs_out' in entry_cfg and entry_cfg['fs_out']:
        return int(entry_cfg['fs_out'])
    if 'f_max' in entry_cfg and entry_cfg['f_max']:
        # Nyquist: fs_out >= 2 * f_max. usamos ceil para inteiro
        return int(math.ceil(2.0 * float(entry_cfg['f_max'])))
    if 'decimation' in entry_cfg and entry_cfg['decimation']:
        dec = float(entry_cfg['decimation'])
        if dec <= 0:
            raise ValueError("decimation deve ser > 0")
        return max(1, int(round(fs_in / dec)))
    # fallback conservador: metade da taxa de entrada
    return max(1, int(fs_in // 2))

def transformar_e_salvar(
    arquivo: str,
    out_root: str,
    sensor: str,
    condicao: str,
    entry_cfg: dict,
    overwrite: bool = False,
):
    """
    Mantém a estrutura de pastas original relativa a raw_root
    """
    # Caminho relativo a partir de raw_root
    rel_path = Path(arquivo).relative_to(Path(out_root).parent / "raw")
    
    # Caminho de saída mantendo a estrutura
    dest_path = Path(out_root) / rel_path
    
    # Se já existe e não quer sobrescrever, retorna
    if dest_path.exists() and not overwrite:
        print(f"[pulado] Já existe: {dest_path}")
        return str(dest_path)
    
    # Garante que o diretório de destino existe
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Carrega e processa os dados
    df = carregar_parquet(arquivo)
    fs_in = float(entry_cfg['fs_in'])
    fs_out = inferir_fs_out_entry(entry_cfg, fs_in)
    df_out = reamostrar_dataframe(df, fs_in=fs_in, fs_out=fs_out)
    
    # Salva o arquivo
    df_out.to_parquet(dest_path, index=False)
    
    # Salva metadados
    meta = {
        "origem": str(arquivo),
        "sensor": sensor,
        "condicao": condicao,
        "banda": inferir_banda_de_nome_ou_caminho(arquivo),
        "fs_in": fs_in,
        "fs_out": fs_out,
        "n_in": int(df.shape[0]),
        "n_out": int(df_out.shape[0]),
        "colunas": list(df_out.columns),
    }
    
    # Salva metadados em um arquivo .meta.json ao lado do arquivo
    meta_path = dest_path.with_suffix('.meta.json')
    meta_path.write_text(pd.Series(meta).to_json(indent=2), encoding='utf-8')
    
    print(f"[ok] {arquivo} -> {dest_path} ({meta['n_in']} -> {meta['n_out']} amostras)")
    return str(dest_path)

# --- Função para processar todo o raw e aplicar filtros (include/exclude patterns) ---
def processar_raw_tree(
    raw_root: str,
    out_root: str = None,
    fs_map: dict = None,
    include_patterns: list = None,
    exclude_patterns: list = None,
    overwrite: bool = False,
):
    """
    - raw_root: pasta 'raw' (ex: 'Dados/raw')
    - out_root: se None, cria sibling raw_resampled (ex: 'Dados/raw_resampled')
    - fs_map: dicionário que mapeia por SENSOR/COND/BANDA ou por banda:
         Exemplo mínimo:
         fs_map = {
            'Voltage': {'CAL': {'HF': {'fs_in':10000, 'f_max':3000}, 'LF': {'fs_in':1000, 'decimation':5}}},
            'Current_HIF': {'HF': {'fs_in':10000, 'fs_out':2000}},
            'DEFAULT': {'HF': {'fs_in':10000, 'decimation':5}, 'LF': {'fs_in':1000, 'decimation':2}}
         }
       A função tenta localizar a cfg por sensor->condicao->banda, senão por sensor->banda, senão por banda em 'DEFAULT'.
    - include_patterns: lista de substrings (case-insensitive). Se None -> inclui tudo.
    - exclude_patterns: lista de substrings para pular.
    """
    raw_root = Path(raw_root)
    if out_root is None:
        out_root = raw_root.with_name(raw_root.name + "_resampled")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    include_patterns = [p.lower() for p in (include_patterns or [])]
    exclude_patterns = [p.lower() for p in (exclude_patterns or [])]

    # varre todos .parquet em subpastas
    for p in sorted(raw_root.rglob("*.parquet")):
        spath = str(p).lower()
        # aplicar include/exclude
        if include_patterns:
            if not any(q in spath for q in include_patterns):
                # não casa com nenhum include
                continue
        if exclude_patterns:
            if any(q in spath for q in exclude_patterns):
                continue

        # tentar inferir sensor e condicao a partir da árvore de pastas
        parts = [pp for pp in p.parts]
        # heurística: sensor é a pasta imediatamente abaixo raw_root
        try:
            rel = p.relative_to(raw_root)
            parts_rel = rel.parts
            # exemplo: Voltages_cal/HF/10.parquet  -> sensor = parts_rel[0]; possivelmente condicao = parts_rel[0] tem _
            sensor = parts_rel[0]
            # condicao pode ser parte do nome do sensor (e.g., Currents_cal -> condicao=CAL)
            condicao = "CAL" if "cal" in sensor.lower() else "FAULT"
            # banda: procurar HF/LF nos componentes relativos
            banda = 'unknown'
            for s in parts_rel:
                if s.upper() == 'HF':
                    banda = 'HF'
                    break
                if s.upper() == 'LF':
                    banda = 'LF'
                    break
        except Exception:
            sensor = parts[0] if parts else "unknown"
            condicao = "unknown"
            banda = 'unknown'

        # localizar configuração no fs_map com fallback
        cfg_entry = None
        if fs_map:
            # tentativas decrescentes de chave
            # 1) sensor -> condicao -> banda
            try:
                cfg_entry = fs_map.get(sensor, {}).get(condicao, {}).get(banda)
            except Exception:
                cfg_entry = None
            # 2) sensor -> banda
            if cfg_entry is None:
                try:
                    cfg_entry = fs_map.get(sensor, {}).get(banda)
                except Exception:
                    cfg_entry = None
            # 3) DEFAULT -> banda
            if cfg_entry is None:
                try:
                    cfg_entry = fs_map.get('DEFAULT', {}).get(banda)
                except Exception:
                    cfg_entry = None

        if cfg_entry is None:
            print(f"[ignorado] sem cfg para {p} (sensor={sensor}, condicao={condicao}, banda={banda})")
            continue

        transformar_e_salvar(
            arquivo=str(p),
            out_root=str(out_root),
            sensor=sensor,
            condicao=condicao,
            entry_cfg=cfg_entry,
            overwrite=overwrite,
        )