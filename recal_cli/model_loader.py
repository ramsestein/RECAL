"""
recal_cli.model_loader
=======================
Carga modelos en cualquier formato común y devuelve un objeto con
.predict_proba(X: np.ndarray) -> np.ndarray (probabilidades de clase positiva).

Formatos soportados (auto-detección por extensión):
    .json, .ubj, .bin   → XGBoost
    .joblib, .pkl       → sklearn / cualquier estimador con predict_proba
    .h5, .keras         → Keras / TensorFlow
    .pt, .pth           → PyTorch (state_dict requiere loader BYOM)

BYOM (Bring Your Own Model):
    Si el formato no es nativo, el usuario provee un módulo Python con:
        def load_model(path: str | Path):
            ...
            return obj  # debe tener .predict_proba(X) o ser callable

    Se invoca con --model-loader path/to/loader.py
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class PredictProbaModel(Protocol):
    """Protocolo mínimo: cualquier objeto con predict_proba(X) → ndarray."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


# ── Wrappers para uniformizar la interfaz ─────────────────────────────────────

class _SklearnLikeWrapper:
    """Envuelve estimadores sklearn-like que retornan (n, 2) y devolvemos col[1]."""

    def __init__(self, model, n_features: int | None = None):
        self._model = model
        self.n_features_in_ = n_features or getattr(model, "n_features_in_", None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()


class _XGBoostJSONWrapper:
    """Wrapper for XGBoost loaded from JSON/UBJ/BIN."""

    def __init__(self, booster, n_features: int):
        self._booster = booster
        self.n_features_in_ = n_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        # XGBoost acepta NaN nativamente; convertir Inf → NaN
        X_clean = np.where(np.isinf(X), np.nan, X)
        dmat = xgb.DMatrix(X_clean)
        return self._booster.predict(dmat)

    def feature_importance(self) -> dict:
        """Importancias por gain del booster, alineadas a self.n_features_in_.

        Devuelve un dict {nombre_o_indice: gain}. RECAL solo necesita la
        magnitud relativa, así que las claves no tienen que matchear el
        schema (el caller las mapea por orden si no las encuentra).
        """
        try:
            score = self._booster.get_score(importance_type="gain")
        except Exception:
            score = {}
        if not score:
            # devolver array uniforme como fallback
            return {f"f{i}": 1.0 for i in range(self.n_features_in_)}
        return score

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """SHAP values vía pred_contribs (drop bias column)."""
        import xgboost as xgb
        X_clean = np.where(np.isinf(X), np.nan, X)
        dmat = xgb.DMatrix(X_clean)
        contribs = self._booster.predict(dmat, pred_contribs=True)
        # contribs shape: (n, p+1) — última col es bias
        return contribs[:, :-1]


class _KerasWrapper:
    """Wrapper para modelos Keras (h5/keras)."""

    def __init__(self, model):
        self._model = model
        self.n_features_in_ = int(model.input_shape[-1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Keras no acepta NaN — imputar con 0 (el usuario debe pre-imputar
        # apropiadamente si entrenó con otra estrategia)
        X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out = self._model.predict(X, verbose=0)
        if out.ndim == 2 and out.shape[1] == 2:
            return out[:, 1]
        return out.ravel()


class _TorchModuleWrapper:
    """Wrapper para módulos PyTorch ya instanciados (no state_dicts sueltos)."""

    def __init__(self, model, n_features: int | None = None):
        import torch
        self._torch = torch
        self._model = model.eval()
        self.n_features_in_ = n_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        with self._torch.no_grad():
            t = self._torch.from_numpy(X)
            out = self._model(t).cpu().numpy()
        if out.ndim == 2 and out.shape[1] == 2:
            # logits o probas de 2 clases
            if (out.min() < 0) or (out.max() > 1):
                e = np.exp(out - out.max(axis=1, keepdims=True))
                out = e / e.sum(axis=1, keepdims=True)
            return out[:, 1]
        out = out.ravel()
        # Si parece logit, sigmoidear
        if (out.min() < 0) or (out.max() > 1):
            out = 1.0 / (1.0 + np.exp(-out))
        return out


# ── Loaders por formato ───────────────────────────────────────────────────────

def _load_xgboost(path: Path, schema: list[str] | None = None) -> PredictProbaModel:
    """Carga XGBoost JSON/UBJ/BIN.

    Si se pasa schema, usa el `XGBoostWrapper` interno (ya probado, exactamente
    equivalente al pipeline original con AUROC=0.7756). En caso contrario
    devuelve un wrapper minimalista.
    """
    if schema is not None:
        try:
            from recal.model.xgboost_wrapper import XGBoostWrapper
            wrapper = XGBoostWrapper(schema=schema, model_path=path)
            wrapper.n_features_in_ = len(schema)
            logger.info("XGBoost loaded from %s via XGBoostWrapper (n_features=%d)",
                        path.name, len(schema))
            return wrapper
        except Exception as e:
            logger.warning("XGBoostWrapper not available (%s); using minimal wrapper", e)

    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    n_features = int(booster.num_features())
    logger.info("XGBoost loaded from %s (minimal wrapper, n_features=%d)",
                path.name, n_features)
    return _XGBoostJSONWrapper(booster, n_features)


def _load_joblib(path: Path) -> PredictProbaModel:
    import joblib
    obj = joblib.load(path)
    if hasattr(obj, "predict_proba"):
        logger.info("sklearn-compatible model loaded from %s", path.name)
        return _SklearnLikeWrapper(obj)
    raise TypeError(
        f"Object at {path} has no .predict_proba(). "
        "Use --model-loader for BYOM."
    )


def _load_keras(path: Path) -> PredictProbaModel:
    try:
        import keras
        model = keras.models.load_model(str(path))
    except ImportError:
        from tensorflow import keras  # type: ignore
        model = keras.models.load_model(str(path))
    logger.info("Keras model loaded from %s", path.name)
    return _KerasWrapper(model)


def _load_torch(path: Path) -> PredictProbaModel:
    """
    PyTorch: solo soporta torch.save(modelo_completo).
    Para state_dicts sueltos, usa --model-loader.
    """
    import torch
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        raise ValueError(
            f"{path} contiene un state_dict, no un modelo. "
            "Necesitas un BYOM loader que reconstruya la arquitectura. "
            "Crea un .py con `def load_model(path): ...` y úsalo con --model-loader."
        )
    if not callable(obj):
        raise TypeError(f"Objeto cargado de {path} no es callable.")
    return _TorchModuleWrapper(obj)


def _load_byom(loader_path: Path, model_path: Path) -> PredictProbaModel:
    """Carga un modelo usando un loader Python provisto por el usuario."""
    spec = importlib.util.spec_from_file_location("user_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_model"):
        raise AttributeError(
            f"{loader_path} debe definir `def load_model(path): ...`"
        )

    obj = module.load_model(str(model_path))

    if hasattr(obj, "predict_proba"):
        return _SklearnLikeWrapper(obj) if not hasattr(obj, "n_features_in_") else obj
    if callable(obj):
        # Asumimos que ya devuelve probabilidades 1D
        class _CallableWrapper:
            def __init__(self, fn): self._fn = fn
            def predict_proba(self, X): return np.asarray(self._fn(X)).ravel()
        return _CallableWrapper(obj)

    raise TypeError(
        f"load_model() debe devolver un objeto con .predict_proba(X) o un callable. "
        f"Recibido: {type(obj)}"
    )


# ── API pública ───────────────────────────────────────────────────────────────

_DISPATCH: dict[str, Callable[[Path], PredictProbaModel]] = {
    ".json":   _load_xgboost,
    ".ubj":    _load_xgboost,
    ".bin":    _load_xgboost,
    ".joblib": _load_joblib,
    ".pkl":    _load_joblib,
    ".pickle": _load_joblib,
    ".h5":     _load_keras,
    ".keras":  _load_keras,
    ".pt":     _load_torch,
    ".pth":    _load_torch,
}


def load_model(
    path: str | Path,
    model_type: str | None = None,
    custom_loader: str | Path | None = None,
    schema: list[str] | None = None,
) -> PredictProbaModel:
    """
    Carga un modelo y devuelve un objeto con .predict_proba(X).

    Parameters
    ----------
    path : str | Path
        Ruta al archivo del modelo.
    model_type : str, optional
        Forzar formato ('xgboost', 'sklearn', 'keras', 'torch'). Si None,
        se detecta por extensión.
    custom_loader : str | Path, optional
        Ruta a un .py con `def load_model(path)` para formatos no soportados.
    schema : list[str], optional
        Lista ordenada de features. Solo necesaria para XGBoost (mejora
        nombres en feature_importance). Si se omite, igual funciona.

    Returns
    -------
    PredictProbaModel
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")

    if custom_loader is not None:
        return _load_byom(Path(custom_loader), path)

    if model_type is not None:
        type_map = {
            "xgboost": _load_xgboost,
            "sklearn": _load_joblib,
            "joblib":  _load_joblib,
            "keras":   _load_keras,
            "tf":      _load_keras,
            "torch":   _load_torch,
            "pytorch": _load_torch,
        }
        loader = type_map.get(model_type.lower())
        if loader is None:
            raise ValueError(f"model_type desconocido: {model_type}")
        if loader is _load_xgboost:
            return loader(path, schema=schema)
        return loader(path)

    ext = path.suffix.lower()
    loader = _DISPATCH.get(ext)
    if loader is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. Supported native formats: "
            f"{sorted(_DISPATCH.keys())}. Use custom_loader= for BYOM."
        )
    if loader is _load_xgboost:
        return loader(path, schema=schema)
    return loader(path)
