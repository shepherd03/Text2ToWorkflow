from pydantic import BaseModel

from .action_extractor import ActionExtractor
from .config import Settings, load_settings
from .control_intent_extractor import ControlIntentExtractor
from .llm_client import DeepSeekClient
from .preprocessor import TextPreprocessor
from .rulebook import load_rulebook
from .resource_extractor import ResourceExtractor
from .schema import UTR, UTRValidationReport
from .utr_combiner import UTRCombiner
from .validator import UTRValidator
from .variable_extractor import VariableExtractor


class PipelineOutput(BaseModel):
    utr: UTR
    report: UTRValidationReport
    meta: dict[str, str | bool]


class UTRGenerationPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        rules: dict | None = None,
        rulebook_path: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.rulebook_path = rulebook_path
        effective_rules = rules or load_rulebook(rulebook_path)
        llm_client = DeepSeekClient(self.settings) if self.settings.llm_enabled else None
        self.preprocessor = TextPreprocessor(effective_rules)
        self.action_extractor = ActionExtractor(llm_client, effective_rules)
        self.resource_extractor = ResourceExtractor(llm_client, effective_rules)
        self.control_intent_extractor = ControlIntentExtractor(llm_client, effective_rules)
        self.variable_extractor = VariableExtractor(llm_client, effective_rules)
        self.combiner = UTRCombiner()
        self.validator = UTRValidator()

    def run(self, task_text: str) -> PipelineOutput:
        normalized = self.preprocessor.normalize(task_text)
        cleaned_text = str(normalized["cleaned_text"])
        clauses = [str(item) for item in normalized["clauses"]]
        actions = self.action_extractor.extract(cleaned_text, clauses)
        resources = self.resource_extractor.extract(cleaned_text)
        control_intents = self.control_intent_extractor.extract(cleaned_text)
        variables = self.variable_extractor.extract(cleaned_text)
        utr = self.combiner.combine(actions, resources, control_intents, variables)
        report = self.validator.validate(utr, strict_completeness=self.settings.strict_completeness)
        return PipelineOutput(
            utr=utr,
            report=report,
            meta={
                "llm_provider": "deepseek",
                "llm_enabled": self.settings.llm_enabled,
                "strict_completeness": self.settings.strict_completeness,
                "rulebook_source": "runtime_override" if self.rulebook_path else "default_or_env",
            },
        )
