paidiverpy.config.config_params
===============================

.. py:module:: paidiverpy.config.config_params

.. autoapi-nested-parse::

   Configuration parameters module.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.config.config_params.ConfigParams


Module Contents
---------------

.. py:class:: ConfigParams(config_params: dict[str, str | None])

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   Configuration parameters class.

   :param config_params: The configuration parameters.
                         It should have the following keys:
                         - input_path (str): The input path.
                         - output_path (str): The output path.
                         - metadata_path (str): The metadata path.
                         - metadata_type (str): The metadata type.
                         - track_changes (bool): Whether to track changes.
                         - n_jobs (int): The number of jobs.
   :type config_params: Dict

   :raises ValueError: Invalid configuration parameters.















   ..
       !! processed by numpydoc !!

