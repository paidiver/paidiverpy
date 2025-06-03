paidiverpy.frontend.widgets.utils
=================================

.. py:module:: paidiverpy.frontend.widgets.utils

.. autoapi-nested-parse::

   Utility functions for creating widgets in Panel.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.frontend.widgets.utils.create_title
   paidiverpy.frontend.widgets.utils.is_running_in_panel_server


Module Contents
---------------

.. py:function:: create_title(title_str: str, html_h_tag: int = 1, bold: bool = True) -> panel.pane.HTML

   
   Create a title pane with the given string.

   :param title_str: The title text.
   :type title_str: str
   :param html_h_tag: The HTML heading tag to use (1-6).
   :type html_h_tag: int
   :param bold: Whether to make the title bold.
   :type bold: bool

   :returns: A Panel HTML pane containing the title.
   :rtype: pn.pane.HTML















   ..
       !! processed by numpydoc !!

.. py:function:: is_running_in_panel_server() -> bool

   
   Detect if running inside a Panel server (i.e., interactive app mode).

   :returns: True if running in a Panel server, False otherwise.
   :rtype: bool















   ..
       !! processed by numpydoc !!

