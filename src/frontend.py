import streamlit as st

class FileValidatorUI:
    """
    Classe responsável por gerar a interface de usuário para o validador de arquivos Excel.
    """
    def __init__(self):
        self.set_page_config()

    def set_page_config(self):
        st.set_page_config(page_title="IHS FORECASTER", layout="wide")

    def display_header(self):
        st.title("IHS FORECASTER")

    def upload_file(self):
        # Função para carregar arquivo do usuário
        return st.file_uploader("Carregue seu arquivo aqui", type=["parquet", "csv", "xlsx", "xls"])

    def display_results(self, transformed_df, error):
        # Exibe o resultado da transformação ou erro
        if error:
            st.error(f"Erro na validação ou no processamento: {error}")
        else:
            st.success("Arquivo processado com sucesso!")
            st.dataframe(transformed_df)