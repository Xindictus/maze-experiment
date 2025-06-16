from enum import Enum


class RewardStrategy(str, Enum):
    SHAFTI = "shafti"
    DISTANCE = "distance"
    TIMEOUT = "timeout"
