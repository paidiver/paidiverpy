paidiverpy.models.open_params
=============================

.. py:module:: paidiverpy.models.open_params

.. autoapi-nested-parse::

   Custom parameters dataclasses.

   This module contains the dataclasses for the parameters used in the custom_params module.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.models.open_params.ImageOpenArgsRawPyParams
   paidiverpy.models.open_params.ImageOpenArgsRawParams
   paidiverpy.models.open_params.ImageOpenArgsOpenCVParams
   paidiverpy.models.open_params.ImageOpenArgs


Module Contents
---------------

.. py:class:: ImageOpenArgsRawPyParams(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   Parameters for RawPy postprocessing (rawpy.RawPy.postprocess).
















   ..
       !! processed by numpydoc !!

   .. py:class:: Config

      
      Configuration for the RawPy parameters model.
















      ..
          !! processed by numpydoc !!


.. py:class:: ImageOpenArgsRawParams(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   Parameters for manually loading raw images with specific metadata.

   These parameters are required when the image format is not supported by standard libraries.















   ..
       !! processed by numpydoc !!

.. py:class:: ImageOpenArgsOpenCVParams(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   Parameters for OpenCV image loading.
















   ..
       !! processed by numpydoc !!

.. py:class:: ImageOpenArgs(/, **data: Any)

   Bases: :py:obj:`paidiverpy.utils.base_model.BaseModel`


   
   Wrapper for specifying image format and associated parameters.
















   ..
       !! processed by numpydoc !!

   .. py:method:: validate_params() -> ImageOpenArgs

      
      Validate `params` based on `image_type` and cast to appropriate type.
















      ..
          !! processed by numpydoc !!


