paidiverpy.utils.locals
=======================

.. py:module:: paidiverpy.utils.locals

.. autoapi-nested-parse::

   This module have functions related to package versions.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.utils.locals.get_sys_info
   paidiverpy.utils.locals.show_versions
   paidiverpy.utils.locals.modified_environ


Module Contents
---------------

.. py:function:: get_sys_info() -> list[tuple[str, str]]

   
   Returns system information as a dict.

   :returns: A list of tuples containing system information.
   :rtype: list[tuple[str, str]]















   ..
       !! processed by numpydoc !!

.. py:function:: show_versions(file=sys.stdout, conda=False) -> None

   
   Print the versions of paidiverpy and its dependencies.

   :param file: print to the given file-like object. Defaults to sys.stdout.
   :type file: file-like, optional
   :param conda: format versions to be copy/pasted on a conda environment file (default, False)
   :type conda: bool, optional















   ..
       !! processed by numpydoc !!

.. py:function:: modified_environ(*remove, **update)

   
   Temporarily updates the ``os.environ`` dictionary in-place.

   The ``os.environ`` dictionary is updated in-place so that the modification
   is sure to work in all situations.

   :param remove: Environment variables to remove.
   :param update: Dictionary of environment variables and values to add/update.















   ..
       !! processed by numpydoc !!

