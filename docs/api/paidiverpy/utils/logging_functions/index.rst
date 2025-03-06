paidiverpy.utils.logging_functions
==================================

.. py:module:: paidiverpy.utils.logging_functions

.. autoapi-nested-parse::

   Logging utilities.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.utils.logging_functions.VerboseLevel
   paidiverpy.utils.logging_functions.ColorFormatter


Functions
---------

.. autoapisummary::

   paidiverpy.utils.logging_functions.initialise_logging
   paidiverpy.utils.logging_functions.check_raise_error


Module Contents
---------------

.. py:class:: VerboseLevel

   Bases: :py:obj:`enum.IntEnum`


   
   Verbose levels for logging.
















   ..
       !! processed by numpydoc !!

.. py:class:: ColorFormatter(fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)

   Bases: :py:obj:`logging.Formatter`


   
   Custom formatter to add colors to log messages.
















   ..
       !! processed by numpydoc !!

   .. py:method:: format(record: logging.LogRecord) -> str

      
      Format the log message with color.

      :param record: The log record.
      :type record: logging.LogRecord

      :returns: The formatted log message.
      :rtype: str















      ..
          !! processed by numpydoc !!


.. py:function:: initialise_logging(verbose: int = 2, logger_name: str = 'paidiverpy') -> logging.Logger

   
   Initialise logging configuration.

   :param verbose: Verbose level (0 = NONE, 1 = ERRORS_WARNINGS, 2 = INFO, 3 = DEBUG).
                   Defaults to 2.
   :type verbose: int
   :param logger_name: The name of the logger. Defaults to "paidiverpy".
   :type logger_name: str

   :returns: The logger object.
   :rtype: logging.Logger















   ..
       !! processed by numpydoc !!

.. py:function:: check_raise_error(raise_error: bool, message: str) -> None

   
   Check if an error should be raised and raise it if necessary.

   :param raise_error: Whether to raise an error.
   :type raise_error: bool
   :param message: The error message.
   :type message: str

   :raises ValueError: The error message.















   ..
       !! processed by numpydoc !!

