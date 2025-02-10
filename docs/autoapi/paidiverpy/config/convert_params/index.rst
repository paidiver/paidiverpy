paidiverpy.config.convert_params
================================

.. py:module:: paidiverpy.config.convert_params

.. autoapi-nested-parse::

   Convert layer parameters dataclasses.

   This module contains the dataclasses for the parameters of the convert layer
   functions.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   paidiverpy.config.convert_params.CONVERT_LAYER_METHODS


Classes
-------

.. autoapisummary::

   paidiverpy.config.convert_params.BitParams
   paidiverpy.config.convert_params.ToParams
   paidiverpy.config.convert_params.BayerPatternParams
   paidiverpy.config.convert_params.NormalizeParams
   paidiverpy.config.convert_params.ResizeParams
   paidiverpy.config.convert_params.CropParams


Module Contents
---------------

.. py:class:: BitParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the bit conversion.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: output_bits
      :type:  int
      :value: 8



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: ToParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the channel conversion.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: to
      :type:  str
      :value: 'uint8'



   .. py:attribute:: channel_selector
      :type:  int
      :value: 0



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: BayerPatternParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the Bayer pattern conversion.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: bayer_pattern
      :type:  str
      :value: 'BGGR'



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: NormalizeParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the image normalization.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: min
      :type:  float
      :value: 0



   .. py:attribute:: max
      :type:  float
      :value: 1



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: ResizeParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the image resizing.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: min
      :type:  int
      :value: 256



   .. py:attribute:: max
      :type:  int
      :value: 256



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: CropParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the image cropping.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: x
      :type:  tuple


   .. py:attribute:: y
      :type:  tuple


   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:data:: CONVERT_LAYER_METHODS

