"""Apply arbitrary public OpenCV and scikit-image functions to each image."""

# Values and call signatures are intentionally dynamic in this generic adapter.
# ruff: noqa: ANN401, PERF203, TRY301

import importlib
import logging
from collections.abc import Callable
from typing import Any
import cv2
import numpy as np
import skimage
from dask.distributed import Client
from paidiverpy import Paidiverpy
from paidiverpy.config.config_params import ConfigParams
from paidiverpy.config.configuration import Configuration
from paidiverpy.images_layer import ImagesLayer
from paidiverpy.metadata_parser import MetadataParser

_FUNCTION_ROOTS = {"opencv": cv2, "skimage": skimage}
_REFERENCE_ROOTS = {"cv2": cv2, "skimage": skimage, "numpy": np, "np": np}


class LibraryLayer(Paidiverpy):
    """Apply a configured OpenCV or scikit-image function to every image."""

    def __init__(
        self,
        parameters: dict[str, Any],
        config_params: dict[str, Any] | ConfigParams | None = None,
        config_file_path: str | None = None,
        config: Configuration | None = None,
        metadata: MetadataParser | None = None,
        images: ImagesLayer | None = None,
        paidiverpy: Paidiverpy | None = None,
        step_name: str | None = None,
        client: Client | None = None,
        config_index: int | None = None,
        logger: logging.Logger | None = None,
        raise_error: bool = False,
        verbose: int = 2,
    ):
        super().__init__(
            config_params=config_params,
            config_file_path=config_file_path,
            metadata=metadata,
            config=config,
            images=images,
            paidiverpy=paidiverpy,
            client=client,
            logger=logger,
            raise_error=raise_error,
            verbose=verbose,
        )
        self.step_name = step_name
        self.config_index = self.config.add_step(config_index, parameters, step_class=LibraryLayer)
        self.step_metadata = self._calculate_steps_metadata(self.config.steps[self.config_index])

    def run(self) -> None:
        """Resolve the configured function and apply it to every image."""
        function = self.resolve_function(str(self.step_metadata["library"]), str(self.step_metadata["function"]))
        params = self.decode_value(self.step_metadata.get("params") or {})
        image_param = self.step_metadata.get("image_param")
        result_index = self.step_metadata.get("result_index")
        def apply_function(image_data: np.ndarray[Any, Any], filename: str = "", **_kwargs: Any) -> np.ndarray[Any, Any]:
            try:
                if image_param:
                    if image_param in params:
                        msg = f"Parameter '{image_param}' is reserved for the current image"
                        raise ValueError(msg)
                    result = function(**{image_param: image_data}, **params)
                else:
                    result = function(image_data, **params)
                if result_index is not None:
                    if not isinstance(result, tuple | list):
                        msg = "result_index can only be used when the function returns a tuple or list"
                        raise TypeError(msg)
                    result = result[result_index]
                return self.validate_result(result)
            except Exception as exc:
                target = f"{self.step_metadata['library']}.{self.step_metadata['function']}"
                msg = f"Function '{target}' failed for image '{filename}': {exc}"
                raise RuntimeError(msg) from exc

        images = self.process_images(apply_function, {})
        self.update_image_step(images)

    @staticmethod
    def resolve_function(library: str, path: str) -> Callable[..., Any]:
        """Resolve a public callable below an approved library namespace.

        Args:
            library (str): The library name, e.g., "opencv" or "skimage".
            path (str): The dotted path to the function, e.g., "imgproc.cvtColor".

        Returns:
            Callable[..., Any]: The resolved function.
        """
        if library not in _FUNCTION_ROOTS:
            msg = f"Unsupported library '{library}'"
            raise ValueError(msg)
        value = LibraryLayer._resolve_path(_FUNCTION_ROOTS[library], path)
        if not callable(value):
            msg = f"'{path}' is not callable"
            raise TypeError(msg)
        return value

    @staticmethod
    def decode_value(value: Any) -> Any:
        """Recursively translate ``$ref`` and ``$tuple`` YAML representations.

        Args:
            value (Any): The value to decode.

        Returns:
            Any: The decoded value.
        """
        if isinstance(value, list):
            return [LibraryLayer.decode_value(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
                msg = "$ref must be the only key and its value must be a string"
                raise ValueError(msg)
            root_name, separator, path = value["$ref"].partition(".")
            if not separator or root_name not in _REFERENCE_ROOTS:
                msg = f"Unsupported reference '{value['$ref']}'"
                raise ValueError(msg)
            return LibraryLayer._resolve_path(_REFERENCE_ROOTS[root_name], path)
        if "$tuple" in value:
            if set(value) != {"$tuple"} or not isinstance(value["$tuple"], list):
                msg = "$tuple must be the only key and its value must be a list"
                raise ValueError(msg)
            return tuple(LibraryLayer.decode_value(item) for item in value["$tuple"])
        return {key: LibraryLayer.decode_value(item) for key, item in value.items()}

    @staticmethod
    def validate_result(result: Any) -> np.ndarray[Any, Any]:
        """Ensure that a library call produced an image compatible with Paidiverpy.

        Args:
            result (Any): The result to validate.

        Returns:
            np.ndarray[Any, Any]: The validated image array.
        """
        if not isinstance(result, np.ndarray):
            msg = f"function returned {type(result).__name__}, not a NumPy image array"
            raise TypeError(msg)
        if result.ndim not in (2, 3):
            msg = f"function returned an array with {result.ndim} dimensions; expected 2 or 3"
            raise ValueError(msg)
        if result.size == 0:
            msg = "function returned an empty image"
            raise ValueError(msg)
        if result.dtype.hasobject:
            msg = "function returned an object-dtype array"
            raise TypeError(msg)
        return result

    @staticmethod
    def _resolve_path(root: Any, path: str) -> Any:
        """Resolve a dotted path below a root object, importing modules as needed.

        Args:
            root (Any): The root object to start from.
            path (str): The dotted path to resolve.

        Returns:
            Any: The resolved object.
        """
        parts = path.split(".")
        if not path or any(not part.isidentifier() or part.startswith("_") for part in parts):
            msg = f"Invalid or private symbol path '{path}'"
            raise ValueError(msg)

        value = root
        root_name = root.__name__
        for index, part in enumerate(parts):
            try:
                value = getattr(value, part)
            except AttributeError:
                module_name = ".".join([root_name, *parts[: index + 1]])
                try:
                    value = importlib.import_module(module_name)
                except ModuleNotFoundError as exc:
                    msg = f"Public symbol '{path}' was not found under '{root_name}'"
                    raise ValueError(msg) from exc
        return value
