from frontend import FileValidatorUI
from backend import ProcessDataController, MachineLearningCVPredictions


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
        ml_actions = MachineLearningCVPredictions()

        # Processar os dados
        transformed_df, error_message = controller.process_data(uploaded_file)

        # 1ª Cross Validação
        cv_df = ml_actions.multi_windows_cv(transformed_df)
        eval_df = ml_actions.evaluate_cv(cv_df)
        exo_df = ml_actions.generate_exogenous_holidays(transformed_df)
        ml_actions.applyfit(transformed_df)
        first_table_forecast = ml_actions.save_forecast(
            exogenous_df=exo_df, eval_df=eval_df
        )

        # 2ª Cross validação
        dados_futuros = ml_actions.new_data(transformed_df, first_table_forecast)
        new_h = ml_actions.new_horizon(dados_futuros)
        cv_df_2 = ml_actions.multi_windows_cv(df=dados_futuros, horizon=new_h)
        eval_df_2 = ml_actions.evaluate_cv(cv_df_2)
        exo_df2 = ml_actions.generate_exogenous_holidays(dados_futuros, horizon=new_h)
        ml_actions.applyfit(dados_futuros)
        second_table_forecast = ml_actions.save_forecast(
            exogenous_df=exo_df2, eval_df=eval_df_2, horizon=new_h
        )

        final_output = ml_actions.new_data(first_table_forecast, second_table_forecast)
        final_output = ml_actions.final_format(final_output)

        # Exibir os resultados
        ui.display_results(final_output, error_message)


if __name__ == "__main__":
    main()
