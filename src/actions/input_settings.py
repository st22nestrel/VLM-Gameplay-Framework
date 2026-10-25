"""Input transport selection and independent foreground-focus policy."""

from dataclasses import dataclass

INPUT_METHODS = ("sendinput", "postmessage")


@dataclass(frozen=True)
class InputSettings:
    input_method: str = "sendinput"
    require_focus: bool = True

    def __post_init__(self):
        if self.input_method not in INPUT_METHODS:
            raise ValueError(f"input_method must be one of {INPUT_METHODS}")
        if type(self.require_focus) is not bool:
            raise ValueError("require_focus must be a boolean")

    @classmethod
    def from_config(cls, actions):
        return cls(input_method=actions.get("input_method", "sendinput"),
                   require_focus=actions.get("require_focus", True))
