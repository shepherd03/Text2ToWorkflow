from jsonschema import ValidationError, validate
from pydantic import ValidationError as PydanticValidationError

from .schema import UTR, UTRValidationReport


class UTRValidator:
    def validate(self, utr: UTR, strict_completeness: bool = False) -> UTRValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        schema_valid = self._validate_schema(utr, errors)
        logic_valid = self._validate_logic(utr, errors)
        completeness_valid = self._validate_completeness(utr, errors, warnings, strict_completeness)
        return UTRValidationReport(
            schema_valid=schema_valid,
            logic_valid=logic_valid,
            completeness_valid=completeness_valid,
            errors=errors,
            warnings=warnings,
        )

    def _validate_schema(self, utr: UTR, errors: list[str]) -> bool:
        try:
            schema = UTR.model_json_schema()
            payload = utr.model_dump()
            validate(payload, schema)
            UTR.model_validate(payload)
            return True
        except (ValidationError, PydanticValidationError, Exception) as exc:
            errors.append(f"Schema校验失败: {exc}")
            return False

    def _validate_logic(self, utr: UTR, errors: list[str]) -> bool:
        valid = True
        orders = [action.order for action in utr.actions]
        if orders != list(range(1, len(orders) + 1)):
            errors.append("动作顺序必须从1开始连续递增")
            valid = False
        action_ids = {action.action_id for action in utr.actions}
        for intent in utr.control_intents:
            unknown_ids = [aid for aid in intent.target_actions if aid and aid not in action_ids]
            if unknown_ids:
                errors.append(f"控制意图{intent.intent_id}引用了不存在的动作ID: {unknown_ids}")
                valid = False
        var_ids = [var.var_id for var in utr.variables]
        if len(var_ids) != len(set(var_ids)):
            errors.append("变量ID存在重复")
            valid = False
        return valid

    def _validate_completeness(
        self,
        utr: UTR,
        errors: list[str],
        warnings: list[str],
        strict: bool,
    ) -> bool:
        fields = {
            "actions": len(utr.actions),
            "resources": len(utr.resources),
            "control_intents": len(utr.control_intents),
            "variables": len(utr.variables),
        }
        missing = [name for name, size in fields.items() if size == 0]
        if not missing:
            return True
        message = f"完整性校验提示：以下模块为空 {missing}"
        if strict:
            errors.append(message)
            return False
        warnings.append(message)
        return True
