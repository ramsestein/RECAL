"""
recal_cli.pipeline_preprocessor
================================
Aplica un _pipeline.json (PCA + feature selection) a un DataFrame crudo
para producir las features que el modelo espera.

Formato del pipeline JSON:
{
  "pipeline": {
    "all_features": [...],       // 23 raw input features
    "train_medians": {...},      // medianas para imputación
    "clusters": {                // PCA por cluster
      "1": {
        "variables": [...],
        "scaler_mean": [...],
        "scaler_std": [...],
        "pca_components": [[...]],
        "pca_mean": [...]
      }, ...
    },
    "low_vif_vars": [...],       // 16 features que pasan VIF
    "feature_names_out": [...]   // 39 features ordenadas (23 PCA + 16 raw)
  }
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PipelinePreprocessor:
    """Aplica un pipeline de preprocesamiento (PCA + feature selection)."""

    def __init__(self, pipeline_path: str | Path) -> None:
        self._path = Path(pipeline_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Pipeline not found: {self._path}")

        with open(self._path, encoding="utf-8") as f:
            raw = json.load(f)

        pp = raw.get("pipeline", raw)
        self.all_features: list[str] = pp["all_features"]
        self.train_medians: dict[str, float] = pp["train_medians"]
        self.clusters: dict = pp["clusters"]
        self.low_vif_vars: list[str] = pp["low_vif_vars"]
        self.feature_names_out: list[str] = pp["feature_names_out"]

        logger.info(
            "Pipeline loaded: %d raw → %d PCA + %d low-VIF = %d output features",
            len(self.all_features),
            len(self.clusters),
            len(self.low_vif_vars),
            len(self.feature_names_out),
        )

    @property
    def n_features_in(self) -> int:
        return len(self.all_features)

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names_out)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el pipeline completo a un DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame crudo con columnas originales + columna 'label' (opcional).

        Returns
        -------
        pd.DataFrame
            DataFrame con las features de salida (feature_names_out) + 'label' si existía.
        """
        # 1. Extraer features raw y rellenar NaN con medianas de entrenamiento
        X_raw = pd.DataFrame(index=df.index, dtype=float)
        for feat in self.all_features:
            if feat in df.columns:
                X_raw[feat] = pd.to_numeric(df[feat], errors="coerce")
            else:
                X_raw[feat] = np.nan

        n_missing_raw = int(X_raw.isna().sum().sum())
        for feat in self.all_features:
            if feat in self.train_medians:
                X_raw[feat] = X_raw[feat].fillna(self.train_medians[feat])

        if n_missing_raw:
            logger.info(
                "[pipeline] %d NaN values imputed with train medians", n_missing_raw
            )

        # 2. PCA por cluster
        pca_dfs = []
        for cluster_id in sorted(self.clusters.keys(), key=int):
            cluster = self.clusters[cluster_id]
            variables = cluster["variables"]
            scaler_mean = np.array(cluster["scaler_mean"], dtype=float)
            scaler_std = np.array(cluster["scaler_std"], dtype=float)
            pca_components = np.array(cluster["pca_components"], dtype=float)
            pca_mean = np.array(cluster["pca_mean"], dtype=float)

            X_cluster = X_raw[variables].values.astype(float)
            X_scaled = (X_cluster - scaler_mean) / np.maximum(scaler_std, 1e-10)
            X_pca = X_scaled @ pca_components.T + pca_mean

            comp_name = f"C{int(cluster_id):02d}_PC1"
            pca_dfs.append(pd.DataFrame(X_pca, index=df.index, columns=[comp_name]))

        df_pca = pd.concat(pca_dfs, axis=1)

        # 3. Low-VIF features (raw, sin transformar)
        X_vif = pd.DataFrame(index=df.index, dtype=float)
        for feat in self.low_vif_vars:
            if feat in df.columns:
                X_vif[feat] = pd.to_numeric(df[feat], errors="coerce")
            else:
                X_vif[feat] = np.nan

        # 4. Concatenar y ordenar
        result = pd.concat([df_pca, X_vif], axis=1)
        result = result[self.feature_names_out]

        # 5. Preservar label si existe
        if "label" in df.columns:
            result["label"] = df["label"]

        return result
