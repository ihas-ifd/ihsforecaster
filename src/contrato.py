from pydantic import BaseModel, PositiveFloat, Extra
from typing import Dict


class HistoricData(BaseModel):
    """
    Modelo de dados para o histórico de fornecimento.

    Atributos:
        ds (str): Descrição do histórico de fornecimento.
        unique_id (str): Identificador único das séries temporais.
        y (PositiveFloat): Variável dependente
    """

    ds: str
    unique_id: str
    y: PositiveFloat


class BaselinePivoted(BaseModel):
    praca: str
    turno_g: str

