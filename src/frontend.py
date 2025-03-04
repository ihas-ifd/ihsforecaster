import streamlit as st
import pandas as pd
from io import BytesIO


class FileValidatorUI:
    """
    Classe responsável por gerar a interface de usuário para o validador de arquivos Excel.
    """

    def __init__(self):
        self.set_page_config()

    def set_page_config(self):
        st.set_page_config(page_title="IHS FORECASTER", layout="wide")
        self.apply_styles()

    def apply_styles(self):
        st.markdown(
            """
            <style>
                .main-title {
                    font-size: 36px;
                    font-weight: bold;
                    color: #E30031;
                    text-align: center;
                }
                .upload-box {
                    text-align: center;
                }
                .button-style {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 8px;
                    padding: 10px;
                    width: 100%;
                    text-align: center;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def display_header(self):
        st.markdown(
            "<div class='main-title'>IHS FORECASTER</div>", unsafe_allow_html=True
        )

    def upload_file(self):
        # Função para carregar arquivo do usuário
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Carregue seu arquivo aqui", type=["parquet", "csv", "xlsx", "xls"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return uploaded_file

    def display_results(self, transformed_df, error):
        # Exibe o resultado da transformação ou erro
        if error:
            st.error(f"Erro na validação ou no processamento: {error}")
        else:
            st.success("Arquivo processado com sucesso!")
            st.dataframe(transformed_df)

            # Botão para baixar os resultados como Excel
            self.download_excel(transformed_df)

    def download_excel(self, df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Resultados", index=False)
        output.seek(0)

        st.download_button(
            label="Baixar Resultados em Excel",
            data=output,
            file_name="resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
