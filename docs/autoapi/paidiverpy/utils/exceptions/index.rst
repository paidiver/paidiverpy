paidiverpy.utils.exceptions
===========================

.. py:module:: paidiverpy.utils.exceptions

.. autoapi-nested-parse::

   Exception classes.

   ..
       !! processed by numpydoc !!


Exceptions
----------

.. autoapisummary::

   paidiverpy.utils.exceptions.VariableNotFoundError


Functions
---------

.. autoapisummary::

   paidiverpy.utils.exceptions.raise_value_error


Module Contents
---------------

.. py:exception:: VariableNotFoundError(variable_name: str)

   Bases: :py:obj:`Exception`


   
   Exception raised for when a variable is not found in the dataset.

   :param Exception: The base exception class.
   :type Exception: Exception















   ..
       !! processed by numpydoc !!

.. py:function:: raise_value_error(message: str) -> None

   
   Raise a ValueError with the given message.

   :param message: The message to raise the ValueError with.
   :type message: str















   ..
       !! processed by numpydoc !!

