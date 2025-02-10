paidiverpy.utils.install_packages
=================================

.. py:module:: paidiverpy.utils.install_packages

.. autoapi-nested-parse::

   This module contains functions to check and install dependencies.

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   paidiverpy.utils.install_packages.NUM_CHANNELS_GREY
   paidiverpy.utils.install_packages.NUM_CHANNELS_RGB
   paidiverpy.utils.install_packages.NUM_CHANNELS_RGBA
   paidiverpy.utils.install_packages.NUM_IMAGE_DIMS
   paidiverpy.utils.install_packages.DEFAULT_BITS
   paidiverpy.utils.install_packages.EIGHT_BITS
   paidiverpy.utils.install_packages.SIXTEEN_BITS
   paidiverpy.utils.install_packages.THIRTY_TWO_BITS
   paidiverpy.utils.install_packages.PACKAGE_REGEX


Functions
---------

.. autoapisummary::

   paidiverpy.utils.install_packages.check_and_install_dependencies
   paidiverpy.utils.install_packages.is_package_installed


Module Contents
---------------

.. py:data:: NUM_CHANNELS_GREY
   :value: 2


.. py:data:: NUM_CHANNELS_RGB
   :value: 3


.. py:data:: NUM_CHANNELS_RGBA
   :value: 4


.. py:data:: NUM_IMAGE_DIMS
   :value: 2


.. py:data:: DEFAULT_BITS
   :value: 8


.. py:data:: EIGHT_BITS
   :value: 8


.. py:data:: SIXTEEN_BITS
   :value: 16


.. py:data:: THIRTY_TWO_BITS
   :value: 32


.. py:data:: PACKAGE_REGEX

.. py:function:: check_and_install_dependencies(dependencies: list[str] | None, dependencies_path: str | None) -> None

   
   Check and install dependencies.

   :param dependencies: The dependencies to check and install.
   :type dependencies: Union[List[str], None]
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

