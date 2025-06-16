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

    file_handler: bool = Field(
        False, json_schema_extra={"env": "FILE_HANDLER"}
    )

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

    def get_confluence_creds(self) -> Dict[str, str]:
        """_summary_

        Returns:
            Dict[str, str]: _description_
        """
        # The following is used as a separate variable to avoid
        # noqa: E231 of flake8
        url = f"https://{self.confluence_domain}.atlassian.net"  # noqa: E231

        return {
            "v1": {
                "baseUrl": url,
                "space": self.confluence_doc_space,
                "token": self.confluence_access_token,
                "username": self.confluence_username,
            }
        }
