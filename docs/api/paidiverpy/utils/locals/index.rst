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
   paidiverpy.utils.locals.cli_version
   paidiverpy.utils.locals.pip_version
   paidiverpy.utils.locals.get_version
   paidiverpy.utils.locals.show_versions


Module Contents
---------------

.. py:function:: get_sys_info() -> list[tuple[str, str]]

   
   Returns system information as a dict.

   :returns: A list of tuples containing system information.
   :rtype: list[tuple[str, str]]















   ..
       !! processed by numpydoc !!

.. py:function:: cli_version(cli_name: str) -> str

   
   Get the version of a CLI tool.

   :param cli_name: The name of the CLI tool.
   :type cli_name: str

   :returns: The version of the CLI tool.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: pip_version(pip_name: str) -> str

   
   Get the version of a package installed via pip.

   :param pip_name: The name of the package.
   :type pip_name: str

   :returns: The version of the package.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: get_version(module_name: str) -> str

   
   Get the version of a module.

   :param module_name: The name of the module.
   :type module_name: str

   :returns: The version of the module.
   :rtype: str















   ..
       !! processed by numpydoc !!

.. py:function:: show_versions(file: TextIO = sys.stdout, conda: bool = False) -> None

   
   Print the versions of paidiverpy and its dependencies.

   :param file: The file to write the versions to. Defaults to sys.stdout.
   :type file: TextIO, optional
   :param conda: Whether to format the output for conda. Defaults to False.
   :type conda: bool, optional















   ..
       !! processed by numpydoc !!

