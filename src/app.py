from frontend import FileValidatorUI
from backend import ProcessDataController



def main():
    # Instanciando a UI para o validador
    ui = FileValidatorUI()

    # Exibindo o cabeçalho
    ui.display_header()

    # Carregar o arquivo do usuário
    uploaded_file = ui.upload_file()

    if uploaded_file is not None:
        # Criar uma instância do controlador para processar os dados
        controller = ProcessDataController()

        # Processar os dados
        transformed_df, error_message = controller.process_data(uploaded_file)

        # Exibir os resultados
        ui.display_results(transformed_df, error_message)


if __name__ == "__main__":
    main()
