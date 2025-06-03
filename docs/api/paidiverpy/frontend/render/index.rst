paidiverpy.frontend.render
==========================

.. py:module:: paidiverpy.frontend.render

.. autoapi-nested-parse::

   WidgetRenderer class for rendering widgets based on configuration parameters.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.frontend.render.WidgetRenderer


Module Contents
---------------

.. py:class:: WidgetRenderer(steps: bool = False, step_parameters: paidiverpy.utils.base_model.BaseModel | None = None)

   
   Class for rendering widgets based on configuration parameters.

   :param steps: If True, the renderer will handle step-specific widgets.
   :type steps: bool
   :param step_parameters: Parameters for the step if applicable.
   :type step_parameters: dict, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: create_widget(name: str, field: dict, html_h_tag: int = 2) -> panel.Column

      
      Create a widget based on the field type and name.

      :param name: The name of the field.
      :type name: str
      :param field: The field definition containing type, description, default value, etc.
      :type field: dict
      :param html_h_tag: The HTML heading tag level for the title.
      :type html_h_tag: int

      :returns: A Panel Column containing the title and the input widget.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_optional_widget(name: str, field: dict, options: list[str], default: str | float | bool | None = None, html_h_tag: int = 2) -> panel.Column

      
      Create a widget for an optional field.

      :param name: The name of the field.
      :type name: str
      :param field: The field definition containing type, description, default value, etc.
      :type field: dict
      :param options: The options for the union type.
      :type options: list[str]
      :param default: The default value for the field.
      :type default: str | float | bool | None
      :param html_h_tag: The HTML heading tag level for the title.
      :type html_h_tag: int

      :returns: A Panel Column containing the title and the input widget.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: render_custom_types(model_class: str, prefix: str | None = None, html_h_tag: int = 2) -> panel.Column

      
      Render custom types based on the model class and field definition.

      :param model_class: The name of the model class.
      :type model_class: str
      :param field: The field definition if applicable.
      :type field: dict, optional
      :param prefix: The prefix for the field name.
      :type prefix: str, optional
      :param html_h_tag: The HTML heading tag level for the title.
      :type html_h_tag: int

      :returns: A Panel Column containing the rendered widgets.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: render_list_input(field: dict, name: str, html_h_tag: int = 2) -> panel.Column

      
      Render a list input widget based on the field definition.

      :param field: The field definition containing type, description, default value, etc.
      :type field: dict
      :param name: The name of the field.
      :type name: str
      :param html_h_tag: The HTML heading tag level for the title.
      :type html_h_tag: int

      :returns: A Panel Column containing the list input widget.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: render_union_input(selected_type: str, field: dict, name: str, default: str | float | bool | None = None, provide: bool = True, html_h_tag: int = 2) -> panel.widgets.Widget

      
      Render the input widget for a union type field.

      :param selected_type: The selected type from the union.
      :type selected_type: str
      :param field: The field definition containing type, description, default value, etc.
      :type field: dict
      :param name: The name of the field.
      :type name: str
      :param default: The default value for the field.
      :type default: str | float | bool | None
      :param provide: Whether to provide the input widget or not.
      :type provide: bool
      :param html_h_tag: The HTML heading tag level for the title.
      :type html_h_tag: int

      :returns: The input widget for the selected type.
      :rtype: pn.widgets.Widget















      ..
          !! processed by numpydoc !!


   .. py:method:: get_input_widget(type_: str, field: dict, name: str, default: str | float | bool | None = None, html_h_tag: int = 2, provide: bool = True) -> panel.widgets.Widget | None

      
      Get the input widget based on the type and field definition.

      :param type_: The type of the field.
      :type type_: str
      :param field: The field definition containing type, description, default value, etc.
      :type field: dict
      :param name: The name of the field.
      :type name: str
      :param default: The default value for the field.
      :type default: str | float | bool | None
      :param html_h_tag: The HTML heading tag level for the title.
      :type html_h_tag: int
      :param provide: Whether to provide the input widget or not.
      :type provide: bool

      :returns: The input widget for the field, or None if not applicable.
      :rtype: pn.widgets.Widget | None















      ..
          !! processed by numpydoc !!


   .. py:method:: render_method_with_mode_params(field_meta: dict, model_class: str, prefix: str | None = None, html_h_tag: int = 2, widgets: list[panel.Column] | None = None) -> list[panel.Column]

      
      Render the method with mode and parameters based on the field metadata.

      :param field_meta: The field metadata containing mode and parameters.
      :type field_meta: dict
      :param model_class: The name of the model class.
      :type model_class: str
      :param prefix: The prefix for the field name.
      :type prefix: str | None
      :param html_h_tag: The HTML heading tag level for the title.
      :type html_h_tag: int
      :param widgets: Existing widgets to append to.
      :type widgets: list[pn.Column] | None

      :returns: A list of Panel Columns containing the rendered widgets.
      :rtype: list[pn.Column]















      ..
          !! processed by numpydoc !!


   .. py:method:: widget_from_literal(field_name: str, literal_type: paidiverpy.models.sampling_params.Literal[option1, option2, option3]) -> panel.widgets.Select

      
      Create a widget from a literal type.

      :param field_name: The name of the field.
      :type field_name: str
      :param literal_type: The literal type to create the widget from.
      :type literal_type: Literal

      :returns: A Panel Select widget with options from the literal type.
      :rtype: pn.widgets.Select















      ..
          !! processed by numpydoc !!


