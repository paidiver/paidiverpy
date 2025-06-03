paidiverpy.sampling_layer
=========================

.. py:module:: paidiverpy.sampling_layer

.. autoapi-nested-parse::

   
   __init__.py for resample_layer module.
















   ..
       !! processed by numpydoc !!


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/paidiverpy/sampling_layer/sampling_layer/index


Classes
-------

.. autoapisummary::

   paidiverpy.sampling_layer.SamplingLayer


Package Contents
----------------

.. py:class:: SamplingLayer(parameters: dict, config_params: dict | paidiverpy.config.config_params.ConfigParams = None, config_file_path: str | None = None, config: paidiverpy.config.configuration.Configuration = None, metadata: paidiverpy.metadata_parser.MetadataParser = None, images: paidiverpy.images_layer.ImagesLayer = None, paidiverpy: paidiverpy.Paidiverpy = None, step_name: str | None = None, client: dask.distributed.Client | None = None, config_index: int | None = None, add_new_step: bool = True, logger: logging.Logger | None = None, raise_error: bool = False, verbose: int = 2)

   Bases: :py:obj:`paidiverpy.Paidiverpy`


   
   Process the images in the resample layer.

   :param parameters: The parameters for the step.
   :type parameters: dict
   :param config_params: The configuration parameters.
                         It can contain the following keys / attributes:
                         - input_path (str): The path to the input files.
                         - output_path (str): The path to the output files.
                         - metadata_path (str): The path to the metadata file.
                         - metadata_type (str): The type of the metadata file.
                         - track_changes (bool): Whether to track changes.
                         - n_jobs (int): The number of n_jobs.
   :type config_params: dict | ConfigParams, optional
   :param config_file_path: The path to the configuration file.
   :type config_file_path: str
   :param config: The configuration object.
   :type config: Configuration
   :param metadata: The metadata object.
   :type metadata: MetadataParser
   :param images: The images object.
   :type images: ImagesLayer
   :param paidiverpy: The paidiverpy object.
   :type paidiverpy: Paidiverpy
   :param step_name: The name of the step.
   :type step_name: str
   :param client: The Dask client.
   :type client: Client
   :param config_index: The index of the configuration.
   :type config_index: int
   :param add_new_step: Whether to add a new step.
   :type add_new_step: bool
   :param logger: The logger object.
   :type logger: logging.Logger
   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param verbose: verbose level (0 = none, 1 = errors/warnings, 2 = info).
   :type verbose: int















   ..
       !! processed by numpydoc !!

   .. py:method:: run() -> None

      
      Run the resample layer steps on the images based on the configuration.

      Run the resample layer steps on the images based on the configuration.

      :raises ValueError: The mode is not defined in the configuration file.















      ..
          !! processed by numpydoc !!


