"""Safe public validation boundary shared by Praxis SDK models."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.config import ExtraValues

from praxis.exceptions import ModelValidationError


class SafeBaseModel(BaseModel):
    """Convert public constructor/model_validate failures into input-free exceptions.

    Third-party raw Pydantic `TypeAdapter` calls are outside the Praxis SDK validation boundary.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    def __init__(self, /, **data: Any) -> None:
        validation_failed = False
        try:
            super().__init__(**data)
        except ValidationError:
            validation_failed = True
        if validation_failed:
            raise ModelValidationError("模型输入验证失败")

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate Python data without retaining rejected input in exceptions."""

        validated: Self | None = None
        validation_failed = False
        try:
            validated = super().model_validate(
                obj,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except ValidationError:
            validation_failed = True
        if validation_failed or validated is None:
            raise ModelValidationError("模型输入验证失败")
        return validated

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate JSON without retaining rejected input in exceptions."""

        validated: Self | None = None
        validation_failed = False
        try:
            validated = super().model_validate_json(
                json_data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except ValidationError:
            validation_failed = True
        if validation_failed or validated is None:
            raise ModelValidationError("模型输入验证失败")
        return validated

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate string data without retaining rejected input in exceptions."""

        validated: Self | None = None
        validation_failed = False
        try:
            validated = super().model_validate_strings(
                obj,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except ValidationError:
            validation_failed = True
        if validation_failed or validated is None:
            raise ModelValidationError("模型输入验证失败")
        return validated
