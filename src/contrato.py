from pydantic import BaseModel, PositiveFloat

class HistoricoSupply(BaseModel):
    """
    Modelo de dados para o histórico de fornecimento.

    Atributos:
        ds (str): Descrição do histórico de fornecimento.
        unique_id (str): Identificador único das séries temporais.
        supply_seconds (PositiveFloat): Duração do supply em segundos.
    """
    ds: str
    unique_id: str
    supply_seconds: PositiveFloat
  

