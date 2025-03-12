import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend import FileValidatorUI

ui = FileValidatorUI()

ui.display_header("DESPIVOTADOR DE BASELINE")


uploaded_file = ui.upload_file()