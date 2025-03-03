import pandas as pd
from contrato import HistoricoSupply
from pydantic import ValidationError
from itertools import product

class FileLoader:
    def __init__(self):
        self.dataframe = pd.DataFrame()

    def load(self, uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                self.dataframe = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                self.dataframe = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                self.dataframe = pd.read_parquet(uploaded_file)
            else:
                raise ValueError("Formato de arquivo não suportado. Por favor, carregue um arquivo CSV ou Excel.")
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
                    field = error.get('loc', ['unknown'])[0]
                    message = error.get('msg', 'Erro desconhecido')
                    self.errors.append(f"Erro na linha {index + 2}, campo '{field}': {message}")
            except Exception as e:
                self.errors.append(f"Erro inesperado na linha {index + 2}: {str(e)}")
        
        if self.errors:
            return None, self.errors

        return dataframe, None

class DataTransformer:
    def __init__(self):
        pass

    def filling_gaps(self, df):
        unique_dates = df["ds"].unique()
        unique_ids = df["unique_id"].unique()

        full_combinations = pd.DataFrame(product(unique_dates, unique_ids), columns=["ds", "unique_id"])

        df = full_combinations.merge(df, on=["unique_id", "ds"], how="left")

        return df

    def rebuilding_target(self, df):

        df["y"] = df["supply_seconds"] / 3600
        df = df.drop(columns="supply_seconds").fillna(0)

        return df
    
    def transform(self, df):

        df = self.filling_gaps(df)
        df = self.rebuilding_target(df)

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