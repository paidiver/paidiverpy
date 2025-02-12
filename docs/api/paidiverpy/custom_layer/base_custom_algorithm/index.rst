paidiverpy.custom_layer.base_custom_algorithm
=============================================

.. py:module:: paidiverpy.custom_layer.base_custom_algorithm

.. autoapi-nested-parse::

   Base class for custom algorithms.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.custom_layer.base_custom_algorithm.BaseCustomAlgorithm


Module Contents
---------------

.. py:class:: BaseCustomAlgorithm(image_data: numpy.ndarray | dask.array.core.Array, params: paidiverpy.config.custom_params.CustomParams)

   
   Base class for custom algorithms.

   :param image_data: The image data to process
   :type image_data: np.ndarray | dask.array.core.Array
   :param params: The parameters for the custom algorithm
   :type params: DynamicConfig















   ..
       !! processed by numpydoc !!

   .. py:method:: process() -> numpy.ndarray | dask.array.core.Array

      
      Process the image data.

      :returns: The processed image data
      :rtype: np.ndarray | dask.array.core.Array















      ..
          !! processed by numpydoc !!


