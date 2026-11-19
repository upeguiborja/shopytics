"""Semilla y columnas del conjunto de datos de intención de compra."""

SEMILLA = 42
NUMERICAS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
]
LOG_VARS = NUMERICAS
NOMINALES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
BINARIAS = ["Weekend"]
ESPECIALES = ["SpecialDay"]
PREDICTORAS = NUMERICAS + NOMINALES + BINARIAS + ESPECIALES
OBJETIVO = "Revenue"
