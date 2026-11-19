"""Mallas de hiperparámetros para la búsqueda por familia de modelos."""

from .models import construir_modelos


def modelos_y_mallas(
    incluir_pagevalues=True, incluir_specialday=True, codificar_specialday="numerico", excluir=None,
):
    """Construye los pipelines y las mallas base de cada familia."""
    modelos = construir_modelos(
        incluir_pagevalues=incluir_pagevalues,
        incluir_specialday=incluir_specialday,
        codificar_specialday=codificar_specialday,
        excluir=excluir,
    )
    mallas = {}
    mallas["Regresion_logistica"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"],
    }
    mallas["kNN"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__n_neighbors": [5, 15, 31, 45, 61, 81, 101],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }
    mallas["RandomForest"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__max_depth": [None, 12, 20, 30],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_features": ["sqrt", 0.5],
        "clf__class_weight": [None, "balanced"],
    }
    mallas["MLP"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "clf__alpha": [0.0001, 0.01],
        "clf__learning_rate_init": [0.001, 0.01, 0.03, 0.1],
    }
    mallas["SVM"] = [
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["linear"],
            "clf__C": [0.1, 1, 10, 30, 100],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1, 10, 30, 100],
            "clf__gamma": ["scale", 0.01, 0.001],
            "clf__class_weight": [None, "balanced"],
        },
    ]
    return modelos, mallas


def mallas_ampliadas(construir_modelos_fn=modelos_y_mallas, rapido: bool = False):
    """Devuelve (modelos, mallas) con rejillas ampliadas.

    En modo rapido se reduce el numero de valores para pruebas de humo.
    """
    modelos, mallas = construir_modelos_fn()
    if rapido:
        mallas["RandomForest"] = {
            "pre__num__log__usar_log": [True],
            "clf__max_depth": [12],
            "clf__min_samples_leaf": [1],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": [None],
        }
        mallas["MLP"] = {
            "pre__num__log__usar_log": [True],
            "clf__hidden_layer_sizes": [(64,)],
            "clf__alpha": [1e-4],
            "clf__learning_rate_init": [0.01],
        }
        mallas["SVM"] = [
            {"pre__num__log__usar_log": [True], "clf__kernel": ["linear"],
             "clf__C": [1], "clf__class_weight": [None]},
            {"pre__num__log__usar_log": [True], "clf__kernel": ["rbf"],
             "clf__C": [10], "clf__gamma": [0.01], "clf__class_weight": [None]},
        ]
        mallas["kNN"] = {
            "pre__num__log__usar_log": [True],
            "clf__n_neighbors": [61],
            "clf__weights": ["distance"],
            "clf__p": [2],
        }
        mallas["Regresion_logistica"] = {
            "pre__num__log__usar_log": [True],
            "clf__C": [0.1],
            "clf__class_weight": [None],
        }
        return modelos, mallas

    # RF: log(1+x) es monotona y no altera particiones de arboles; no se
    # explora usar_log para evitar candidatos duplicados en el top-k.
    mallas["RandomForest"] = {
        "pre__num__log__usar_log": [True],
        "clf__max_depth": [4, 6, 8, 10, 12, 20, None],
        "clf__max_features": ["log2", "sqrt", 0.3, 0.5, 0.7],
        "clf__min_samples_leaf": [1, 5, 10, 20],
        "clf__class_weight": [None, "balanced"],
    }
    mallas["MLP"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
        "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "clf__learning_rate_init": [0.001, 0.01, 0.03, 0.1, 0.3],
    }
    mallas["SVM"] = [
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["linear"],
            "clf__C": [0.01, 0.03, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1, 10, 30, 100],
            "clf__gamma": ["scale", 0.001, 0.01, 0.03, 0.1],
            "clf__class_weight": [None, "balanced"],
        },
    ]
    return modelos, mallas
