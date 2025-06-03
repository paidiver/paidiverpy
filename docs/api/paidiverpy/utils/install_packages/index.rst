paidiverpy.utils.install_packages
=================================

.. py:module:: paidiverpy.utils.install_packages

.. autoapi-nested-parse::

   This module contains functions to check and install dependencies.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.utils.install_packages.check_and_install_dependencies
   paidiverpy.utils.install_packages.is_package_installed


Module Contents
---------------

.. py:function:: check_and_install_dependencies(dependencies: str | None, dependencies_path: str | None) -> None

   
   Check and install dependencies.

   :param dependencies: The dependencies to check and install.
   :type dependencies: str, None
   :param dependencies_path: The path to the dependencies file.
   :type dependencies_path: str, None

   :raises PackageNotFoundError: If the package is not found.















   ..
       !! processed by numpydoc !!

.. py:function:: is_package_installed(package_name: str) -> bool

   
   Check if the package is installed.

   :param package_name: The package name.
   :type package_name: str

   :returns: Whether the package is installed.
   :rtype: bool















   ..
       !! processed by numpydoc !!

