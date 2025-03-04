import pandas as pd
import numpy as np
from contrato import HistoricoSupply
from pydantic import ValidationError
from itertools import product
from workalendar.america import Brazil
import yaml


class FileLoader:
    def __init__(self):
        self.dataframe = pd.DataFrame()

    def load(self, uploaded_file):
        try:
            if uploaded_file.name.endswith(".csv"):
                self.dataframe = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(
                ".xls"
            ):
                self.dataframe = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                self.dataframe = pd.read_parquet(uploaded_file)
            else:
                raise ValueError(
                    "Formato de arquivo não suportado. Por favor, carregue um arquivo CSV ou Excel."
                )
        except Exception as e:
            return None, f"Erro ao carregar o arquivo: {str(e)}"

        return self.dataframe, None


class DataFrameValidator:
    def __init__(self):
        self.errors = []

    def validate(self, dataframe):
        self.errors = []
        extra_cols = set(dataframe.columns) - set(HistoricoSupply.model_fields.keys())
        if extra_cols:
            return None, f"Colunas extras detectadas: {', '.join(extra_cols)}"

        for index, row in dataframe.iterrows():
            try:
                _ = HistoricoSupply(**row.to_dict())
            except ValidationError as ve:
                for error in ve.errors():
                    field = error.get("loc", ["unknown"])[0]
                    message = error.get("msg", "Erro desconhecido")
                    self.errors.append(
                        f"Erro na linha {index + 2}, campo '{field}': {message}"
                    )
            except Exception as e:
                self.errors.append(f"Erro inesperado na linha {index + 2}: {str(e)}")

        if self.errors:
            return None, self.errors

        return dataframe, None


class HolidayMarker:
    def __init__(self, country_calendar=None):
        # Usa o calendário fornecido ou o padrão (Brasil)
        self.calendar = country_calendar if country_calendar else Brazil()

    def get_holidays_for_years(self, years):
        # Calcula os feriados para os anos fornecidos e retorna um conjunto de datas
        feriados = {
            (pd.to_datetime(date), name)
            for year in years
            for date, name in self.calendar.holidays(year)
        }
        return {date for date, name in feriados}

    def mark_holidays(self, df, date_column="ds"):
        # Converte a coluna de datas para datetime
        df[date_column] = pd.to_datetime(df[date_column])

        # Obtém os anos únicos do dataframe
        anos_unicos = df[date_column].dt.year.unique()

        # Calcula as datas de feriados
        datas_feriados = self.get_holidays_for_years(anos_unicos)

        # Marca as colunas de Natal e Ano Novo
        df["is_christmas"] = np.where(
            (df[date_column].dt.month == 12) & (df[date_column].dt.day == 25), 1, 0
        )
        df["is_new_year"] = np.where(
            (df[date_column].dt.month == 1) & (df[date_column].dt.day == 1), 1, 0
        )

        # Marca os feriados
        df["is_holiday"] = df[date_column].isin(datas_feriados).astype(int)

        return df


class DataTransformer:
    def __init__(self, holiday_marker=None):
        self.holiday_marker = holiday_marker if holiday_marker else HolidayMarker()

    def filling_gaps(self, df):
        unique_dates = df["ds"].unique()
        unique_ids = df["unique_id"].unique()

        full_combinations = pd.DataFrame(
            product(unique_dates, unique_ids), columns=["ds", "unique_id"]
        )

        df = (
            full_combinations.merge(df, on=["unique_id", "ds"], how="left")
            .fillna(0)
            .sort_values(by=["ds", "unique_id"])
        )

        return df

    def transform(self, df):
        df = self.filling_gaps(df)

        if self.holiday_marker:
            df = self.holiday_marker.mark_holidays(df, "ds")

        return df


class ProcessDataController:
    def __init__(self):
        self.file_loader = FileLoader()
        self.dataframe_validator = DataFrameValidator()
        self.data_transformer = DataTransformer()

    def process_data(self, uploaded_file):
        dataframe, load_error = self.file_loader.load(uploaded_file)
        if load_error:
            return None, load_error

        validated_df, validation_error = self.dataframe_validator.validate(dataframe)
        if validation_error:
            return None, validation_error

        # Aplicando transformações ao DataFrame validado
        transformed_df = self.data_transformer.transform(validated_df)

        return transformed_df, None
