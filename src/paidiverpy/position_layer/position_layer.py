""" Open raw image file
"""

from paidiverpy import Paidiverpy


class PositionLayer(Paidiverpy):
    def __init__(
        self,
        config_file_path=None,
        input_path=None,
        output_path=None,
        metadata_path=None,
        metadata_type=None,
        metadata=None,
        config=None,
    ):
        super().__init__(
            config_file_path,
            input_path,
            output_path,
            metadata_path,
            metadata_type,
            metadata,
            config,
        )
