paidiverpy.frontend.widgets.app
===============================

.. py:module:: paidiverpy.frontend.widgets.app

.. autoapi-nested-parse::

   Paidiverpy App: Interactive Pipeline Builder and Image Processor.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.frontend.widgets.app.App


Module Contents
---------------

.. py:class:: App

   
   Main application class for the Paidiverpy frontend.
















   ..
       !! processed by numpydoc !!

   .. py:method:: create_pipeline_widget() -> None

      
      Create the pipeline widget to display the current pipeline configuration.
















      ..
          !! processed by numpydoc !!


   .. py:method:: create_modal(title: str = '', information: str = '', on_cancel: bool = False, on_confirm: collections.abc.Callable | None = None, visible: bool = False) -> panel.Column

      
      Create a modal dialog for confirmation actions.

      :param title: The title of the modal.
      :type title: str
      :param information: The information to display in the modal.
      :type information: str
      :param on_cancel: Whether to attach a cancel action.
      :type on_cancel: bool
      :param on_confirm: A callback function for confirmation action.
      :type on_confirm: callable, optional
      :param visible: Whether the modal should be visible initially.
      :type visible: bool

      :returns: A Panel Column containing the modal dialog.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: update_modal(title: str, information: str, on_confirm: collections.abc.Callable | None = None, on_cancel: collections.abc.Callable | None = None) -> None

      
      Update the modal dialog with new content and callbacks.

      :param title: The new title for the modal.
      :type title: str
      :param information: The new information message for the modal.
      :type information: str
      :param on_confirm: A callback function for confirmation action.
      :type on_confirm: callable, optional
      :param on_cancel: A callback function for cancellation action.
      :type on_cancel: callable, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: update_alert(information: str = '', title: str = '', alert_type: str = 'success', visible: bool = True) -> None

      
      Update the alert widget with a message.

      :param information: The information message to display.
      :type information: str
      :param title: The title of the alert.
      :type title: str
      :param alert_type: The type of alert (e.g., "success", "danger").
      :type alert_type: str
      :param visible: Whether the alert should be visible.
      :type visible: bool















      ..
          !! processed by numpydoc !!


   .. py:method:: create_pipeline_functionality() -> None

      
      Create the functionality for running the pipeline.
















      ..
          !! processed by numpydoc !!


   .. py:method:: update_images() -> None

      
      Update the images widget with the processed images from the pipeline.
















      ..
          !! processed by numpydoc !!


   .. py:method:: update_images_widget() -> None

      
      Update the images widget to display the processed images.
















      ..
          !! processed by numpydoc !!


   .. py:method:: create_template() -> panel.template.BootstrapTemplate

      
      Create the main application template with sidebar and main content.

      :returns: The main application template.
      :rtype: pn.template.BootstrapTemplate















      ..
          !! processed by numpydoc !!


   .. py:method:: confirm_general_update(widgets: panel.Column) -> None

      
      Confirm the update of the general configuration.

      :param widgets: The widgets containing the general configuration inputs.
      :type widgets: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_general_widget() -> paidiverpy.frontend.widgets.config_general.AppGeneral

      
      Create the general configuration widget to manage the pipeline's general settings.

      :returns: An instance of the AppGeneral class containing the general configuration widget.
      :rtype: AppGeneral















      ..
          !! processed by numpydoc !!


   .. py:method:: create_steps_widget() -> panel.Column

      
      Create the steps widget to manage the steps in the pipeline.

      :returns: A Panel Column containing the steps configuration form.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_steps_form() -> panel.Column

      
      Create the steps form for adding or updating steps in the pipeline.

      :returns: A Panel Column containing the steps form.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_form(step_number: int, step_parameters: dict | None = None) -> panel.Column

      
      Create a form for adding or updating a step in the pipeline.

      :param step_number: The step number for the form.
      :type step_number: int
      :param step_parameters: Optional parameters for the step if updating.
      :type step_parameters: dict | None

      :returns: A Panel Column containing the form for the step.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_images_widget() -> panel.Column

      
      Create the images widget to display the processed images.

      :returns: A Panel Column containing the images output editor.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_code_yaml_widget() -> panel.Column

      
      Create the code and YAML output widget to display the generated configuration and code.

      :returns: A Panel Column containing the YAML and code output editors.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_yaml_widget() -> panel.Column

      
      Create the YAML output widget to display the generated configuration.

      :returns: A Panel Column containing the YAML output editor and export button.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: create_code_widget() -> panel.Column

      
      Create the code output widget to display the generated code.

      :returns: A Panel Column containing the code output editor.
      :rtype: pn.Column















      ..
          !! processed by numpydoc !!


   .. py:method:: get_button_name(expanded: str, title: str) -> str

      
      Get the button name based on the expansion state.

      :param expanded: The key in the `self.expanded` dictionary.
      :type expanded: str
      :param title: The title of the section.
      :type title: str

      :returns: The formatted button name with an arrow indicating expansion state.
      :rtype: str















      ..
          !! processed by numpydoc !!


   .. py:method:: update_general_configuration(json_str: dict) -> bool

      
      Update the general configuration of the pipeline.

      :param json_str: The JSON string containing the general configuration.
      :type json_str: dict

      :returns: True if the configuration was updated successfully, False otherwise.
      :rtype: bool















      ..
          !! processed by numpydoc !!


   .. py:method:: update_step_configuration(json_str: dict, idx: int | None = None) -> None

      
      Update the step configuration in the pipeline.

      :param json_str: The JSON string containing the step configuration.
      :type json_str: dict
      :param idx: The index of the step to update. If None, a new step is added.
      :type idx: int | None















      ..
          !! processed by numpydoc !!


   .. py:method:: show() -> None

      
      Display the application.
















      ..
          !! processed by numpydoc !!


