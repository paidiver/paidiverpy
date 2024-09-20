""" Open raw image file
"""

from paidiverpy import Paidiverpy


class PositionLayer(Paidiverpy):
    def __init__(
        self,
        config_file_path=None,
        input_path=None,
        output_path=None,
        catalog_path=None,
        catalog_type=None,
        catalog=None,
        config=None,
    ):
        super().__init__(
            config_file_path,
            input_path,
            output_path,
            catalog_path,
            catalog_type,
            catalog,
            config,
        )
