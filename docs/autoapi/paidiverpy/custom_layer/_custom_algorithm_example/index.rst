paidiverpy.custom_layer._custom_algorithm_example
=================================================

.. py:module:: paidiverpy.custom_layer._custom_algorithm_example

.. autoapi-nested-parse::

   This is an example of a custom algorithm that scales the image data using MinMaxScaler from sklearn.preprocessing.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.custom_layer._custom_algorithm_example.MyMethod


Module Contents
---------------

.. py:class:: MyMethod(image_data: numpy.ndarray | dask.array.core.Array, params: paidiverpy.config.custom_params.CustomParams)

   Bases: :py:obj:`paidiverpy.custom_layer.base_custom_algorithm.BaseCustomAlgorithm`


   
   This class scales the image data using MinMaxScaler from sklearn.preprocessing.
















   ..
       !! processed by numpydoc !!

   .. py:method:: process() -> numpy.ndarray

      
      This method scales the image data using MinMaxScaler from sklearn.preprocessing.

      :returns: The scaled image data.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


