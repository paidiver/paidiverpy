from unittest.mock import MagicMock
from datetime import datetime, timedelta


class MockFTP(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = {}
        self.cwd_path = "/"
        self.models = ["global-ocean", "shelf-amm7", "shelf-amm15"]
        self.forcast_days = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
        self.model_date = datetime.now()
        self.variables = ["TEM", "SAL", "CUR", "SSH"]
        self.output_type = ["hi", "lo"]
        for model in self.models:
            for forecast_day in self.forcast_days:
                for variable in self.variables:
                    for output_type in self.output_type:
                        file_name = ""
                        if model == "global-ocean":
                            file_name += f"/{model}/metoffice_coupled_orca025_GL4_"
                        elif model == "shelf-amm7":
                            file_name += f"/{model}/metoffice_foam1_amm7_NWS_"
                        elif model == "shelf-amm15":
                            file_name += f"/{model}/metoffice_foam1_amm15_NWS_"
                        model_date_run = self.model_date + timedelta(days=forecast_day)
                        file_name += (
                            f'{variable}_b{self.model_date.strftime("%Y%m%d")}_'
                        )
                        file_name += f"{output_type}"
                        file_name += f'{model_date_run.strftime("%Y%m%d")}.nc'
                        self.add_file(file_name, b"file content")

    def login(self, user, passwd):
        pass  # Simulate login, do nothing in the mock

    def cwd(self, path):
        self.cwd_path = path

    def nlst(self, path):
        full_path = path
        return [file for file in self.files.keys() if file.startswith(full_path)]

    def retrbinary(self, cmd, callback):
        file_data = self.files.get(cmd.split(" ")[1], b"")
        callback(file_data)

    def storebinary(self, cmd, file):
        self.files[cmd.split(" ")[1]] = file.read()

    def add_file(self, path, content):
        self.files[path] = content
