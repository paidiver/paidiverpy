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

.. py:class:: BaseCustomAlgorithm(image_data: numpy.ndarray | dask.array.core.Array, metadata: dict, params: paidiverpy.config.custom_params.CustomParams, metadata_core: pandas.DataFrame)

   
   Base class for custom algorithms.

   :param image_data: The image data to process
   :type image_data: np.ndarray | dask.array.core.Array
   :param metadata: The metadata for the image data
   :type metadata: dict
   :param params: The parameters for the custom algorithm
   :type params: CustomParams
   :param metadata_object: The metadata object for the image data.
   :type metadata_object: pd.DataFrame, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: process() -> numpy.ndarray | dask.array.core.Array

      
      Process the image data.

      :returns: The processed image data
      :rtype: np.ndarray | dask.array.core.Array















      ..
          !! processed by numpydoc !!


