paidiverpy.config.colour_params
===============================

.. py:module:: paidiverpy.config.colour_params

.. autoapi-nested-parse::

   Colour layer parameters dataclasses.

   This module contains the dataclasses for the parameters of the colour layer
   functions.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   paidiverpy.config.colour_params.COLOUR_LAYER_METHODS


Classes
-------

.. autoapisummary::

   paidiverpy.config.colour_params.GrayScaleParams
   paidiverpy.config.colour_params.GaussianBlurParams
   paidiverpy.config.colour_params.EdgeDetectionParams
   paidiverpy.config.colour_params.SharpenParams
   paidiverpy.config.colour_params.ContrastAdjustmentParams
   paidiverpy.config.colour_params.IlluminationCorrectionParams
   paidiverpy.config.colour_params.DeblurParams
   paidiverpy.config.colour_params.ColourAlterationParams


Module Contents
---------------

.. py:class:: GrayScaleParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the grayscale conversion.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: keep_alpha
      :type:  bool
      :value: False



   .. py:attribute:: method
      :type:  str
      :value: 'opencv'



   .. py:attribute:: invert_colours
      :type:  bool
      :value: False



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: GaussianBlurParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the Gaussian blur.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: sigma
      :type:  float
      :value: 1.0



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: EdgeDetectionParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the edge detection.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: method
      :type:  str
      :value: 'sobel'



   .. py:attribute:: blur_radius
      :type:  float
      :value: 1.0



   .. py:attribute:: threshold
      :type:  float
      :value: 0.1



   .. py:attribute:: object_type
      :type:  str
      :value: 'bright'



   .. py:attribute:: object_selection
      :type:  str
      :value: 'largest'



   .. py:attribute:: estimate_sharpness
      :type:  bool
      :value: False



   .. py:attribute:: deconv
      :type:  bool
      :value: False



   .. py:attribute:: deconv_method
      :type:  str
      :value: 'LR'



   .. py:attribute:: deconv_iter
      :type:  int
      :value: 10



   .. py:attribute:: deconv_mask_weight
      :type:  float
      :value: 0.03



   .. py:attribute:: small_float_val
      :type:  float
      :value: 1e-06



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: SharpenParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the sharpening.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: alpha
      :type:  float
      :value: 1.5



   .. py:attribute:: beta
      :type:  float


   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: ContrastAdjustmentParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the contrast adjustment.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: method
      :type:  str
      :value: 'clahe'



   .. py:attribute:: kernel_size
      :type:  int
      :value: None



   .. py:attribute:: clip_limit
      :type:  float
      :value: 0.01



   .. py:attribute:: gamma_value
      :type:  float
      :value: 0.5



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: IlluminationCorrectionParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the illumination correction.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: method
      :type:  str
      :value: 'rolling'



   .. py:attribute:: radius
      :type:  int
      :value: 100



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: DeblurParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the deblurring.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: method
      :type:  str
      :value: 'wiener'



   .. py:attribute:: psf_type
      :type:  str
      :value: 'gaussian'



   .. py:attribute:: sigma
      :type:  float
      :value: 20



   .. py:attribute:: angle
      :type:  int
      :value: 45



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:class:: ColourAlterationParams

   Bases: :py:obj:`paidiverpy.utils.dynamic_classes.DynamicConfig`


   
   This class contains the parameters for the colour alteration.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: method
      :type:  str
      :value: 'white_balance'



   .. py:attribute:: raise_error
      :type:  bool
      :value: False



.. py:data:: COLOUR_LAYER_METHODS

