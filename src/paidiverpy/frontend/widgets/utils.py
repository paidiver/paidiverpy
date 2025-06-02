"""Utility functions for creating widgets in Panel."""

import panel as pn


def create_title(title_str: str, html_h_tag: int = 1, bold: bool = True) -> pn.pane.HTML:
    """Create a title pane with the given string.

    Args:
        title_str (str): The title text.
        html_h_tag (int): The HTML heading tag to use (1-6).
        bold (bool): Whether to make the title bold.

    Returns:
        pn.pane.HTML: A Panel HTML pane containing the title.
    """
    class_name = f"ppy-pn-title-{html_h_tag}"
    if bold:
        class_name += " ppy-pn-bold"
    return pn.pane.HTML(f"<div class='{class_name}'>{title_str}</div>")


def is_running_in_panel_server() -> bool:
    """Detect if running inside a Panel server (i.e., interactive app mode).

    Returns:
        bool: True if running in a Panel server, False otherwise.
    """
    return pn.state.curdoc and pn.state.curdoc.session_context is not None
