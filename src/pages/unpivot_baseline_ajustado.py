import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend import FileValidatorUI
from backend import ProcessDataController
from contrato import BaselinePivoted


def main():
    ui = FileValidatorUI()

    ui.display_header("DESPIVOTADOR DE BASELINE")


    uploaded_file = ui.upload_file()

    if uploaded_file is not None:
      
        controller = ProcessDataController(BaselinePivoted)
        transformed_df, error_message = controller.process_data(uploaded_file)

        ui.display_results(transformed_df, error_message)


if __name__ == "__main__":
    main()