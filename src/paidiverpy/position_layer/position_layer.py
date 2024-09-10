""" Open raw image file
"""

import os
import shutil
from datetime import datetime
import glob
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.distance import geodesic
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
import geopandas as gpd
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
