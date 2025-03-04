import pandas as pd
import numpy as np
from contrato import HistoricoSupply
from pydantic import ValidationError
from itertools import product
from workalendar.america import Brazil
import yaml
import random
from numba import njit
from window_ops.rolling import rolling_mean
from window_ops.shift import shift_array
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    RollingMean,
    ExponentiallyWeightedMean,
    SeasonalRollingMean,
)
from mlforecast.target_transforms import LocalStandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os
from utilsforecast.losses import rmse


class Naive(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag1"]


class FirstFctHorizon:
    h = 14


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


class ModelParamsLoader:
    """
    Responsável por carregar os parâmetros do modelo a partir de um arquivo YAML.
    """

    def __init__(self, config_path="ml_params.yaml"):
        """
        Inicializa a classe com o caminho do arquivo de configuração.

        :param config_path: Caminho do arquivo YAML de parâmetros do modelo.
        """
        # Garante que o caminho seja absoluto
        self.config_path = self._get_absolute_path(config_path)
        self.params = self._load_params()

    def _get_absolute_path(self, config_path):
        """
        Retorna o caminho absoluto do arquivo YAML, independentemente do diretório de execução.

        :param config_path: Caminho do arquivo YAML (relativo ou absoluto).
        :return: Caminho absoluto do arquivo YAML.
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_directory, config_path)

    def _load_params(self):
        """
        Carrega os parâmetros do modelo a partir do arquivo YAML.

        :return: Parâmetros do modelo ou um dicionário vazio em caso de erro.
        """
        try:
            with open(self.config_path, "r") as file:
                params = yaml.safe_load(file)
                if not params:
                    print(f"⚠️ O arquivo {self.config_path} está vazio.")
                return params or {}
        except FileNotFoundError:
            print(f"⚠️ Arquivo {self.config_path} não encontrado.")
        except yaml.YAMLError as e:
            print(f"⚠️ Erro ao carregar YAML: {e}")
        except Exception as e:
            print(f"⚠️ Erro inesperado: {e}")
        return {}

    def get(self, model_name):
        """
        Obtém os parâmetros de um modelo específico.

        :param model_name: Nome do modelo.
        :return: Parâmetros do modelo ou um dicionário vazio.
        """
        params = self.params.get(model_name)
        if not params:
            print(f"⚠️ Parâmetros para o modelo '{model_name}' não encontrados.")
        return params or {}


# Definir a função de pré-processamento
class ModelPreprocessingFunctions:
    def __init__(self):
        pass

    @staticmethod
    def year_index(times):
        return times.year

    @staticmethod
    def month_index(times):
        return times.month

    @staticmethod
    def day_index(times):
        return times.day

    @staticmethod
    def month_start_or_end(dates):
        """Verifica se a data é o início ou o fim do mês"""
        return dates.is_month_start | dates.is_month_end

    @staticmethod
    def is_sunday(times):
        """Verifica se a data é domingo"""
        return times.dayofweek == 6

    @staticmethod
    @njit
    def rolling_mean_21(x):
        return rolling_mean(x, window_size=21)

    @staticmethod
    @njit
    def ratio_over_previous(x, offset=1):
        """Calcula a razão entre o valor atual e o valor de `offset` de atraso"""
        ratio = x / shift_array(x, offset=offset)
        if offset < 30:
            ratio[np.isinf(ratio)] = 1
            ratio[np.isnan(ratio)] = 1
        else:
            choice = random.randint(0, 1)
            ratio[np.isinf(ratio)] = 0.9 if choice == 0 else 1.1
            ratio[np.isnan(ratio)] = 0.9 if choice == 0 else 1.1
        return ratio

    @staticmethod
    @njit
    def diff_over_previous(x, offset=1):
        """Calcula a diferença entre o valor atual e o valor de `offset` de atraso"""
        return x - shift_array(x, offset=offset)


# Agora, dentro de ModelApplyPreprocessing, usamos essas funções ao configurar o MLForecast
class ModelApplyPreprocessing:
    def __init__(self, config_path="ml_params.yaml"):
        self.params_loader = ModelParamsLoader(config_path)  # Carregar parâmetros
        self.models = self._initialize_models()
        self.preprocessing = ModelPreprocessingFunctions()
        self.fcst = self._initialize_fcst()

    def _initialize_models(self):
        params = self.params_loader.params  # Carregar os parâmetros

        models = {
            "naive": Naive(),
            "lgbm": LGBMRegressor(**params.get("lgbm", {}), verbosity=-1),
            "xbg": XGBRegressor(**params.get("xbg", {})),
            "lasso": Lasso(**params.get("lasso", {})),
            "lin_reg": LinearRegression(),
            "ridge": Ridge(**params.get("ridge", {})),
            "knn": KNeighborsRegressor(**params.get("knn", {})),
            "rf": RandomForestRegressor(**params.get("rf", {})),
            "gb": GradientBoostingRegressor(**params.get("gb", {})),
            "hgb": HistGradientBoostingRegressor(**params.get("hgb", {})),
        }
        return models

    def _initialize_fcst(self):
        # Aqui vamos configurar o MLForecast com o pré-processamento
        fcst = MLForecast(
            models=self.models,
            freq="D",
            target_transforms=[LocalStandardScaler()],
            lags=[1, 7, 14, 21, 28, 35, 42, 49],
            lag_transforms={
                1: [
                    ExpandingMean(),
                    self.preprocessing.ratio_over_previous,
                    (self.preprocessing.ratio_over_previous, 7),
                ],
                2: [self.preprocessing.diff_over_previous],
                7: [
                    RollingMean(window_size=7),
                    self.preprocessing.diff_over_previous,
                    (self.preprocessing.ratio_over_previous, 7),
                    SeasonalRollingMean(7, 7),
                ],
                14: [
                    RollingMean(window_size=7),
                    self.preprocessing.diff_over_previous,
                    (self.preprocessing.ratio_over_previous, 7),
                    SeasonalRollingMean(7, 7),
                ],
                28: [
                    RollingMean(window_size=7),
                    self.preprocessing.diff_over_previous,
                    (self.preprocessing.ratio_over_previous, 7),
                    SeasonalRollingMean(7, 7),
                ],
            },
            date_features=[
                self.preprocessing.year_index,
                self.preprocessing.month_index,
                self.preprocessing.day_index,
                self.preprocessing.is_sunday,
                self.preprocessing.month_start_or_end,
            ],
        )
        return fcst

    def get_fcst(self):
        # Retorna o objeto 'fcst' para ser usado em outras classes
        return self.fcst

    def preprocessing_pipe(self):
        # Método que retorna o MLForecast configurado com pré-processamento
        return self.fcst


class ProcessDataController:
    def __init__(self):
        self.file_loader = FileLoader()
        self.dataframe_validator = DataFrameValidator()
        self.data_transformer = DataTransformer()
        self.data_preparation = ModelApplyPreprocessing()

    def process_data(self, uploaded_file):
        dataframe, load_error = self.file_loader.load(uploaded_file)
        if load_error:
            return None, load_error

        validated_df, validation_error = self.dataframe_validator.validate(dataframe)
        if validation_error:
            return None, validation_error

        # Aplicando transformações ao DataFrame validado
        transformed_df = self.data_transformer.transform(validated_df)
        # fcst = self.data_preparation.preprocessing_pipe()
        # prep_df = fcst.preprocess(transformed_df, static_features=[])

        return transformed_df, None


class MachineLearningCVPredictions:
    def __init__(self, holiday_marker=None):
        self.model_preprocessing = ModelApplyPreprocessing()
        self.fcst = self.model_preprocessing.get_fcst()  # Obtendo o fcst já configurado
        self.holiday_marker = holiday_marker if holiday_marker else HolidayMarker()

    def multi_windows_cv(self, df, n_windows=20, horizon=FirstFctHorizon.h):
        cv_df = None  # Garantindo que cv_df comece como None
        while n_windows > 0:
            try:
                # Tentativa de fazer a validação cruzada
                cv_df = self.fcst.cross_validation(
                    df=df.copy(),
                    h=horizon,
                    n_windows=n_windows,
                    refit=False,
                    static_features=[],
                )
                print(f"Cross-validation successful with {n_windows} windows!")
                break  # Se rodou sem erro, paramos o loop
            except Exception as e:
                print(f"Failed with {n_windows} windows. Trying {n_windows - 1}...")
                n_windows -= 1  # Reduz uma janela e tenta de novo

        # Verificando se o cv_df foi atribuído corretamente
        if n_windows == 0 and cv_df is None:
            print("No valid number of windows found.")
        else:
            self.cv_df = cv_df  # Atribuindo somente se o cv_df não for None
        return cv_df

    def evaluate_cv(self, df):
        # Realizando a avaliação apenas se o df não estiver vazio ou None
        models = df.columns.drop(["unique_id", "ds", "y", "cutoff"]).tolist()
        evals = rmse(df, models=models)
        evals["best_model"] = evals[models].idxmin(axis=1)

        return evals

    def generate_exogenous_holidays(self, df, horizon=FirstFctHorizon.h):
        # Verificando se o DataFrame original tem as colunas necessárias
        if "ds" not in df.columns or "unique_id" not in df.columns:
            print(
                "As colunas 'ds' e 'unique_id' são obrigatórias e estão faltando no DataFrame original!"
            )
            return

        # 1. Gerar as datas de início e fim para as variáveis exógenas
        start_date = df.ds.max() + pd.Timedelta(days=1)
        end_date = df.ds.max() + pd.Timedelta(days=horizon)

        # 2. Gerar o intervalo de datas para as variáveis exógenas
        exogenous_df = pd.Series(pd.date_range(start=start_date, end=end_date))

        exogenous_df = exogenous_df.to_frame()
        exogenous_df.columns = ["ds"]  # Definindo o nome da coluna de datas como 'ds'

        # 3. Gerar combinações de datas com os unique_ids
        unique_dates = exogenous_df["ds"].unique()
        unique_ids = df["unique_id"].unique()

        exogenous_df = pd.DataFrame(
            product(unique_dates, unique_ids), columns=["ds", "unique_id"]
        )

        # 4. Mesclar com o DataFrame original, sem a coluna 'y'
        exogenous_df = pd.merge(
            exogenous_df, df.drop(columns="y"), on=["ds", "unique_id"], how="left"
        )

        # 5. Marcar feriados se o holiday_marker estiver disponível
        if self.holiday_marker:
            exogenous_df = self.holiday_marker.mark_holidays(exogenous_df, "ds")

        return exogenous_df

    def applyfit(self, df):
        self.fcst.fit(df=df, static_features=[])

    def save_forecast(self, exogenous_df=None, eval_df=None, horizon=FirstFctHorizon.h):
        df_first_forecast = self.fcst.predict(h=horizon, X_df=exogenous_df)

        df_first_forecast = df_first_forecast.merge(
            eval_df[["unique_id", "best_model"]]
        )

        df_first_forecast["best_model_result"] = df_first_forecast.apply(
            lambda row: row[row["best_model"]], axis=1
        )

        df_first_forecast = df_first_forecast[["ds", "unique_id", "best_model_result"]]

        df_first_forecast.columns = ["ds", "unique_id", "y"]

        return df_first_forecast

    def new_data(self, df, first_forecast):
        new_table = pd.concat(
            [df[["ds", "unique_id", "y"]], first_forecast[["ds", "unique_id", "y"]]],
            ignore_index=True,
        )

        return new_table

    def new_horizon(self, new_table):
        new_h = (
            (new_table.ds.max() + pd.offsets.MonthEnd(0)) - new_table.ds.max()
        ).days
        return new_h

    def final_format(self, final_output):
        final_output["y"] = np.where(final_output["y"] < 0, 0, final_output["y"])
        final_output[["praca", "turno_g"]] = final_output["unique_id"].str.extract(
            r"^(.*?)\|\|(.*)$"
        )
        final_output = final_output[["ds", "praca", "turno_g", "y"]]
        final_output["ds"] = final_output["ds"].dt.date

        return final_output
