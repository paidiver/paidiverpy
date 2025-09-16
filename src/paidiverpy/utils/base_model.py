"""Dynamic classes for configuration."""

import warnings
from pathlib import Path
from typing import Any
from pydantic import BaseModel as PydanticBaseModel

# Suppress all Pydantic serializer warnings about dict input
warnings.filterwarnings("ignore", category=UserWarning, message="Pydantic serializer warnings:\n  Expected `.*` but got `dict`")


class BaseModel(PydanticBaseModel):
    """Base model for dynamic configurations."""

    def to_dict(self, convert_path: bool = True) -> dict[str, Any]:
        """Convert model to dictionary, excluding None and empty values."""
        raw_dict = self.model_dump(exclude_none=True)
        return {k: str(v) if convert_path and isinstance(v, str | Path) else v for k, v in raw_dict.items() if v not in (None, {}, [])}
