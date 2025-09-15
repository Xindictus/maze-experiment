from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """_summary_

    Args:
        BaseSettings (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Default settings will be validated
    model_config = SettingsConfigDict(validate_default=True)

    # Define PY script folder
    project_dir: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parent.parent)
    )

    file_handler: bool = Field(True, json_schema_extra={"env": "FILE_HANDLER"})

    def get_settings(self) -> Dict[str, any]:
        """_summary_

        Returns:
            Dict[str, any]: _description_
        """

        # Settings to exclude
        settings_to_exclude = {}

        # Filtered dictionary
        filtered_settings = {
            k: v
            for k, v in self.model_dump().items()
            if k not in settings_to_exclude
        }

        return filtered_settings
