paidiverpy.frontend.json_dump
=============================

.. py:module:: paidiverpy.frontend.json_dump

.. autoapi-nested-parse::

   This module provides functions to extract values from a Panel layout and convert them into a structured JSON-like dictionary.

   ..
       !! processed by numpydoc !!


Functions
---------

.. autoapisummary::

   paidiverpy.frontend.json_dump.find_deep_layout
   paidiverpy.frontend.json_dump.check_valid_inputs
   paidiverpy.frontend.json_dump.parse_name
   paidiverpy.frontend.json_dump.insert_nested
   paidiverpy.frontend.json_dump.extract_values
   paidiverpy.frontend.json_dump.extract_json


Module Contents
---------------

.. py:function:: find_deep_layout(layout: panel.widgets.Widget, founds: list[panel.widgets.Widget]) -> list[panel.widgets.Widget]

   
   Recursively find all widgets and layouts in a Panel layout.

   :param layout: The Panel layout or widget to search.
   :type layout: pn.widgets.Widget
   :param founds: A list to collect found widgets and layouts.
   :type founds: list[pn.widgets.Widget]

   :returns: A list of found widgets and layouts.
   :rtype: list[pn.widgets.Widget]















   ..
       !! processed by numpydoc !!

.. py:function:: check_valid_inputs(widget: panel.widgets.Widget, step: bool = False) -> bool

   
   Check if a widget is valid for extraction.

   :param widget: The widget to check.
   :type widget: pn.widgets.Widget
   :param step: If True, additional checks for step widgets are applied.
   :type step: bool

   :returns: True if the widget is valid, False otherwise.
   :rtype: bool















   ..
       !! processed by numpydoc !!

.. py:function:: parse_name(name: str) -> list

   
   Parse a widget name into a list of keys.

   :param name: The name of the widget, which may contain dots and brackets.
   :type name: str

   :returns: A list of keys parsed from the name, converting numeric parts to integers.
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: insert_nested(result: dict, keys: list, value: any) -> None

   
   Insert a value into a nested dictionary structure based on keys.

   :param result: The dictionary to insert into.
   :type result: dict
   :param keys: A list of keys indicating the path to insert the value.
   :type keys: list
   :param value: The value to insert.
   :type value: any















   ..
       !! processed by numpydoc !!

.. py:function:: extract_values(widgets: list[panel.widgets.Widget], step: bool = False) -> dict

   
   Extract values from a list of widgets and return them as a structured dictionary.

   :param widgets: A list of Panel widgets to extract values from.
   :type widgets: list[pn.widgets.Widget]
   :param step: If True, the extraction will consider step-specific widgets.
   :type step: bool

   :returns: A dictionary containing the extracted values, structured by widget names.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: extract_json(layout: panel.widgets.Widget, step: bool = False) -> dict

   
   Extract JSON-like dictionary from a Panel layout or widget.

   :param layout: The Panel layout or widget to extract from.
   :type layout: pn.widgets.Widget
   :param step: If True, the extraction will consider step-specific widgets.
   :type step: bool

   :returns: A dictionary containing the extracted values, structured by widget names.
   :rtype: dict















   ..
       !! processed by numpydoc !!

