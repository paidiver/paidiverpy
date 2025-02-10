paidiverpy.config.config_params
===============================

.. py:module:: paidiverpy.config.config_params

.. autoapi-nested-parse::

   Configuration parameters module.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   paidiverpy.config.config_params.REQUIRED_KEYS


Classes
-------

.. autoapisummary::

   paidiverpy.config.config_params.ConfigParams


Module Contents
---------------

.. py:data:: REQUIRED_KEYS
   :value: ['input_path', 'output_path', 'metadata_path', 'metadata_type', 'track_changes', 'n_jobs']


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

   .. py:attribute:: config_params


   .. py:attribute:: input_path


   .. py:attribute:: output_path


   .. py:attribute:: metadata_path


   .. py:attribute:: metadata_type


   .. py:attribute:: track_changes


   .. py:attribute:: n_jobs


   .. py:method:: _validate_config_params(config_params: dict[str, str | None]) -> dict[str, str | None]

      
      Validate the configuration parameters.

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

      :returns: The validated configuration parameters.
      :rtype: Dict















      ..
          !! processed by numpydoc !!


