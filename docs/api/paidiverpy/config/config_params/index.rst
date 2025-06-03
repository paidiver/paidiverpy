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

.. py:class:: ConfigParams(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   Configuration parameters using Pydantic.

   Fields:
       input_path (Path): The input path.
       output_path (Path): The output path.
       image_open_args (str): The image type.
       metadata_path (Path): The metadata path.
       metadata_type (str): The metadata type.
       track_changes (bool): Whether to track changes. Defaults to True.
       n_jobs (int): Number of jobs. Defaults to 1.















   ..
       !! processed by numpydoc !!

   .. py:method:: validate_required_keys(values: dict) -> dict
      :classmethod:


      
      Validate the required keys in the configuration parameters.

      :param values: The values to validate.
      :type values: dict

      :raises ValueError: If any of the required keys are missing.

      :returns: The validated values.
      :rtype: dict















      ..
          !! processed by numpydoc !!


