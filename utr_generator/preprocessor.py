import re


class TextPreprocessor:
    def __init__(self, rules: dict) -> None:
        section = rules.get("preprocessor", {})
        self.replacement_map = section.get("replacement_map", {})
        split_keywords = section.get("split_keywords", [])
        escaped = [re.escape(item) for item in split_keywords]
        self.split_pattern = rf"(?:{'|'.join(escaped)})" if escaped else r"(?:然后|同时)"

    def clean(self, text: str) -> str:
        content = text.strip()
        content = re.sub(r"\s+", " ", content)
        content = content.replace("；", "，").replace("。", "，")
        content = re.sub(r"[!！]+", "，", content)
        content = re.sub(r"[?？]+", "，", content)
        content = re.sub(r"，{2,}", "，", content).strip("，")
        for src, dst in self.replacement_map.items():
            content = content.replace(src, dst)
        return content

    def split_clauses(self, text: str) -> list[str]:
        segments = re.split(self.split_pattern, text)
        cleaned = [part.strip(" ，,") for part in segments if part.strip(" ，,")]
        return cleaned if cleaned else [text]

    def normalize(self, text: str) -> dict[str, list[str] | str]:
        cleaned = self.clean(text)
        clauses = self.split_clauses(cleaned)
        return {"cleaned_text": cleaned, "clauses": clauses}
