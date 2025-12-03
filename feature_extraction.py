from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from dados import iterar_dataset
try:  # opcional
    from scipy.signal import welch
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
try:  # opcional
    from stockwell import st
    HAS_STOCKWELL = True
except Exception:
    HAS_STOCKWELL = False


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else np.nan


def _bandpower_fft(x: np.ndarray, fs: Optional[float], fmax: Optional[float]) -> float:
    if fs is None or fmax is None or x.size == 0:
        return np.nan
    x = x.astype(float)
    n = x.size
    xf = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / (fs * n)
    mask = xf <= float(fmax)
    return float(np.trapz(psd[mask], xf[mask])) if mask.any() else np.nan


def _bandpower_range_fft(x: np.ndarray, fs: Optional[float], fmin: float, fmax: float) -> float:
    if fs is None or x.size == 0:
        return np.nan
    x = x.astype(float)
    n = x.size
    xf = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / (fs * n)
    mask = (xf >= float(fmin)) & (xf <= float(fmax))
    return float(np.trapz(psd[mask], xf[mask])) if mask.any() else np.nan


def _bandpower_range_welch(x: np.ndarray, fs: Optional[float], fmin: float, fmax: float) -> float:
    if not HAS_SCIPY or fs is None or x.size == 0:
        return _bandpower_range_fft(x, fs, fmin, fmax)
    f, Pxx = welch(x.astype(float), fs=fs, nperseg=min(len(x), 1024))
    mask = (f >= float(fmin)) & (f <= float(fmax))
    return float(np.trapz(Pxx[mask], f[mask])) if mask.any() else np.nan


def _harmonic_bandpowers(
    x: np.ndarray,
    fs: Optional[float],
    base_freq: float = 50.0,
    n_harm: int = 20,
    bw_hz: float = 2.0,
    method: str = "welch",
    individual: bool = False,
) -> Dict[str, float]:
    if fs is None or x.size == 0 or base_freq <= 0 or n_harm <= 0:
        return {}
    out: Dict[str, float] = {}
    acc = []

    if method.lower() == "stockwell" and HAS_STOCKWELL:
        # S-transform: potência aproximada somando |S|^2 em bandas
        S = st.st(x.astype(float))  # shape (n_f, n_t)
        # eixo de frequência (Hz) para S-transform (aprox. 0..Nyquist)
        n = x.size
        f = np.linspace(0, fs / 2, S.shape[0])
        for k in range(1, n_harm + 1):
            fc = k * base_freq
            fmin, fmax = max(0.0, fc - bw_hz), fc + bw_hz
            mask = (f >= fmin) & (f <= fmax)
            p = float(np.mean(np.abs(S[mask, :]) ** 2)) if mask.any() else np.nan
            acc.append(p)
            if individual:
                # out[f"hp_{int(fc)}"] = p
                out[f"h_{k}"] = p
    else:
        # Welch por padrão; fallback para FFT
        for k in range(1, n_harm + 1):
            fc = k * base_freq
            fmin, fmax = max(0.0, fc - bw_hz), fc + bw_hz
            p = _bandpower_range_welch(x, fs, fmin, fmax)
            acc.append(p)
            if individual:
                # out[f"hp_{int(fc)}"] = p
                out[f"h_{k}"] = p

    # agregados
    # vals = np.array([v for v in acc if np.isfinite(v)], dtype=float)
    # out["hp_sum"] = float(np.sum(vals)) if vals.size else np.nan
    # out["hp_mean"] = float(np.mean(vals)) if vals.size else np.nan
    return out


def compute_metrics(
    s: pd.Series,
    metrics: Iterable[str],
    fs: Optional[float] = None,
    spec_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    x = s.to_numpy(dtype=float, copy=False)
    out: Dict[str, float] = {}
    use = set(m.lower() for m in metrics)

    if "mean" in use:
        out["mean"] = float(np.mean(x)) if x.size else np.nan
    if "std" in use:
        out["std"] = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    if "min" in use:
        out["min"] = float(np.min(x)) if x.size else np.nan
    if "max" in use:
        out["max"] = float(np.max(x)) if x.size else np.nan
    if "ptp" in use or "peak_to_peak" in use:
        out["ptp"] = float(np.ptp(x)) if x.size else np.nan
    if "rms" in use:
        out["rms"] = _rms(x)
    if "skew" in use:
        m = np.mean(x)
        sd = np.std(x)
        out["skew"] = float(np.mean(((x - m) / sd) ** 3)) if sd > 0 and x.size else np.nan
    if "kurt" in use or "kurtosis" in use:
        m = np.mean(x)
        v = np.var(x)
        out["kurt"] = float(np.mean(((x - m) ** 4)) / (v * v)) if v > 0 and x.size else np.nan

    if spec_cfg:
        # Example: spec_cfg = {"bandpower": {"fmax": 1000}}
        bp = spec_cfg.get("bandpower") if isinstance(spec_cfg, dict) else None
        if bp and ("bandpower" in use or "bp" in use):
            fmax = bp.get("fmax") if isinstance(bp, dict) else None
            out["bandpower"] = _bandpower_fft(x, fs=fs, fmax=fmax)
        # Harmonics up to N*base_freq with bandwidth
        hc = spec_cfg.get("harmonics") if isinstance(spec_cfg, dict) else None
        if hc and ("harmonics" in use or "hp" in use):
            base = float(hc.get("base", 50.0))
            n_h = int(hc.get("n", 20))
            bw = float(hc.get("bw", 2.0))
            method = str(hc.get("method", "welch"))
            individual = bool(hc.get("individual", False))
            out.update(_harmonic_bandpowers(x, fs, base, n_h, bw, method, individual))

    return out


def extract_features_from_df(
    df: pd.DataFrame,
    metrics: Iterable[str],
    signal_col: str = "S",
    fs: Optional[float] = None,
    spec_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    if signal_col not in df.columns:
        return {}
    return compute_metrics(df[signal_col].dropna(), metrics=metrics, fs=fs, spec_cfg=spec_cfg)


def build_feature_dataset(
    base_dir: str | Path,
    output_path: str | Path,
    metrics: Iterable[str] = ("mean", "std", "min", "max", "rms", "ptp"),
    signal_col: str = "S",
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    fs_lookup: Optional[Dict[str, float]] = None,
    spec_cfg: Optional[Dict] = None,
    file_format: str = "parquet",
    flush_every: int = 200,
    limite_por_grupo: Optional[int] = None,
    nrows: Optional[int] = None,
) -> Path:
    """Gera dataset de features a partir de árvore de dados (ex.: Dados/raw_resampled).

    - Usa iterar_dataset(base_dir) para percorrer os arquivos com metadados.
    - Aplica métricas configuráveis sobre a coluna do sinal (padrão: "S").
    - Escreve CSV/Parquet em output_path de forma eficiente (flush em lotes).
    """
    base_dir = Path(base_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    include = [s.lower() for s in include_patterns] if include_patterns else []
    exclude = [s.lower() for s in exclude_patterns] if exclude_patterns else []

    def _match_filters(p: str) -> bool:
        sp = p.lower()
        if include and not any(tok in sp for tok in include):
            return False
        if exclude and any(tok in sp for tok in exclude):
            return False
        return True

    for df, meta in iterar_dataset(
        base_dir=base_dir,
        limite_por_grupo=limite_por_grupo,
        incluir_label=True,
        adicionar_origem=True,
        nrows=nrows,
    ):
        if not _match_filters(meta.get("arquivo", "")):
            continue

        fs = None
        if fs_lookup:
            # chave comum: f"{sensor}/{condicao}/{banda}" ou apenas banda
            key1 = f"{meta.get('sensor')}/{meta.get('condicao')}/{meta.get('banda')}"
            key2 = f"{meta.get('sensor')}/{meta.get('banda')}"
            key3 = meta.get("banda")
            fs = fs_lookup.get(key1) or fs_lookup.get(key2) or fs_lookup.get(key3)

        feats = extract_features_from_df(
            df,
            metrics=metrics,
            signal_col=signal_col,
            fs=fs,
            spec_cfg=spec_cfg,
        )
        if not feats:
            continue

        row = {
            **feats,
            "label": meta.get("label"),
            "sensor": meta.get("sensor"),
            "condicao": meta.get("condicao"),
            "banda": meta.get("banda"),
            "arquivo": meta.get("arquivo"),
            "nome": meta.get("nome"),
        }
        rows.append(row)

        if len(rows) >= flush_every:
            _flush_rows(rows, output_path, file_format)
            rows.clear()

    if rows:
        _flush_rows(rows, output_path, file_format)
        rows.clear()

    return output_path


def _flush_rows(rows: List[Dict], output_path: Path, file_format: str) -> None:
    df = pd.DataFrame.from_records(rows)
    if output_path.exists():
        if file_format == "csv":
            df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            # Parquet append: concatenar ao existente de forma simples
            existing = pd.read_parquet(output_path)
            pd.concat([existing, df], ignore_index=True).to_parquet(output_path, index=False)
    else:
        if file_format == "csv":
            df.to_csv(output_path, index=False)
        else:
            df.to_parquet(output_path, index=False)


__all__ = [
    "compute_metrics",
    "extract_features_from_df",
    "build_feature_dataset",
]

