"""Parametros y resultados CV historicos, no una nueva seleccion automatizada.

MLP (64,) con lr=0.01 fue seleccionado por validacion; el ganador de la
malla por AP-CV fue (64, 32), lr=0.1, con AP-CV 0.7343. La prueba historica
ya fue consultada, por lo que esta reproduccion no es independiente.
"""

from proyecto_ml import SEMILLA

PARAMS = {
    "RandomForest": {
        "clf__class_weight": None, "clf__max_depth": 12,
        "clf__max_features": "sqrt", "clf__min_samples_leaf": 1,
        "clf__n_estimators": 300, "clf__random_state": SEMILLA,
        "clf__n_jobs": 1, "pre__num__log__usar_log": True,
    },
    "MLP": {
        "clf__hidden_layer_sizes": (64,), "clf__alpha": 0.0001,
        "clf__learning_rate_init": 0.01, "clf__solver": "adam",
        "clf__max_iter": 500, "clf__early_stopping": True,
        "clf__validation_fraction": 0.15, "clf__n_iter_no_change": 20,
        "clf__random_state": SEMILLA, "pre__num__log__usar_log": True,
    },
    "SVM": {
        "clf__kernel": "rbf", "clf__C": 10, "clf__gamma": 0.01,
        "clf__class_weight": None, "pre__num__log__usar_log": True,
    },
    "kNN": {
        "clf__n_neighbors": 61, "clf__weights": "distance", "clf__p": 2,
        "pre__num__log__usar_log": True,
    },
    "Regresion_logistica": {
        "clf__C": 0.1, "clf__class_weight": None,
        "pre__num__log__usar_log": True,
    },
}

# Valores archivados de mejores_configuraciones.csv, no recalculados.
CV_HISTORICO = {
    "RandomForest": (0.7531505650864327, 0.03496482251000137),
    "MLP": (0.729354006574404, 0.03911776363446021),
    "SVM": (0.7265851520647685, 0.033579164779107676),
    "kNN": (0.720316, 0.026245),
    "Regresion_logistica": (0.689678974384505, 0.03196829989325587),
}
