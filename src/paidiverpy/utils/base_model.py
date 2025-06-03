"""Dynamic classes for configuration."""

from pathlib import Path
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """Base model for dynamic configurations."""

    def to_dict(self, convert_path: bool = True) -> dict:
        """Convert model to dictionary, excluding None and empty values."""
        raw_dict = self.model_dump(exclude_none=True)
        return {k: str(v) if convert_path and isinstance(v, str | Path) else v for k, v in raw_dict.items() if v not in (None, {}, [])}
