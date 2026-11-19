"""Reductores ajustados exclusivamente sobre las sesiones de entrenamiento."""

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from umap import UMAP

from .utils import SEMILLA


class UMAPSinEtiquetas(UMAP):
    """Impide que Pipeline pase las clases a UMAP durante el ajuste."""

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def pipeline_reducido(modelo, metodo, dimensiones, vecinos=None):
    base = clone(modelo)
    if metodo == "PCA":
        n = int(dimensiones) if dimensiones >= 1 else dimensiones
        reductor = PCA(n_components=n, svd_solver="full")
    elif metodo == "UMAP":
        reductor = UMAPSinEtiquetas(n_components=int(dimensiones), n_neighbors=int(vecinos),
                                    random_state=SEMILLA, n_jobs=1, compatibility_layout=True)
    else:
        raise ValueError(f"Método desconocido: {metodo}")
    return Pipeline([("pre", base["pre"]), ("reduccion", reductor), ("clf", base["clf"])])
