paidiverpy.utils.logging
========================

.. py:module:: paidiverpy.utils.logging

.. autoapi-nested-parse::

   Logging utilities.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.utils.logging.VerboseLevel
   paidiverpy.utils.logging.ColorFormatter


Functions
---------

.. autoapisummary::

   paidiverpy.utils.logging.initialise_logging
   paidiverpy.utils.logging.check_raise_error


Module Contents
---------------

.. py:class:: VerboseLevel

   Bases: :py:obj:`enum.IntEnum`


   
   Verbose levels for logging.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: NONE
      :value: 0



   .. py:attribute:: ERRORS_WARNINGS
      :value: 1



   .. py:attribute:: INFO
      :value: 2



   .. py:attribute:: DEBUG
      :value: 3



.. py:class:: ColorFormatter(fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)

   Bases: :py:obj:`logging.Formatter`


   
   Custom formatter to add colors to log messages.
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: COLORS
      :type:  ClassVar[dict[str, str]]


   .. py:attribute:: RESET
      :type:  ClassVar[str]
      :value: '\x1b[0m'



   .. py:method:: format(record: logging.LogRecord) -> str

      
      Format the log message with color.

      :param record: The log record.
      :type record: logging.LogRecord

      :returns: The formatted log message.
      :rtype: str















      ..
          !! processed by numpydoc !!


.. py:function:: initialise_logging(verbose: int = 2) -> logging.Logger

   
   Initialise logging configuration.

   :param verbose: Verbose level (0 = NONE, 1 = ERRORS_WARNINGS, 2 = INFO, 3 = DEBUG).
                   Defaults to 2.
   :type verbose: int

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

