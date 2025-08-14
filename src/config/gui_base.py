from pydantic import BaseModel, Field


class GUIBaseConfig(BaseModel):
    # The time in secs the goal screen is being displayed
    # TODO: Unused
    goal_screen_display_duration: int = Field(default=3, ge=1, le=10)

    # The time in secs the timeout or goal screen is being displayed
    popup_window_time: int = Field(default=3, ge=1, le=10)

    # The time in secs the set-up screen is being displayed
    start_up_screen_display_duration: int = Field(default=5, ge=1, le=10)

    # The time in secs the timeout screen is being displayed
    # TODO: Unused
    timeout_screen_display_duration: int = Field(default=3, ge=1, le=10)
