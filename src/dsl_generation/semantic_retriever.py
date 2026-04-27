from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from src.core.config import Settings, load_settings
from src.core.schema import DifyNodeType
from src.dsl_generation.node_mapping_rules import MAPPING_RULES_BY_NODE_TYPE


DEFAULT_SEMANTIC_PROTOTYPES: dict[DifyNodeType, list[str]] = {
    DifyNodeType.llm: [
        "generate summarize rewrite analyze translate explain draft polish refine answer respond",
        "summary report email response article content text writing copy narrative",
    ],
    DifyNodeType.code: [
        "transform parse normalize clean validate convert compute filter sort reshape deduplicate",
        "json csv dataset payload schema records fields rows columns values table",
    ],
    DifyNodeType.http_request: [
        "request fetch call invoke query submit trigger sync api endpoint service http webhook url",
        "get post put delete patch external service response payload",
    ],
    DifyNodeType.template_transform: [
        "format compose fill render template assemble structure layout message output notice memo",
        "report email prompt announcement document response",
    ],
    DifyNodeType.tool: [
        "use invoke run search query execute lookup scan tool plugin integration calendar mail",
        "ocr database filesystem automation web crawler",
    ],
    DifyNodeType.doc_extractor: [
        "extract read parse load ingest recognize scan import document pdf file attachment contract invoice",
        "manual paper resume statement policy text",
    ],
    DifyNodeType.parameter_extractor: [
        "extract identify classify detect tag route recognize locate parse map label",
        "parameters entities intents labels slots categories keywords attributes topic region",
    ],
    DifyNodeType.variable_aggregator: [
        "merge aggregate collect combine consolidate join fuse summarize results outputs responses",
        "metrics scores fragments statistics insights branch data",
    ],
    DifyNodeType.iteration: [
        "iterate loop for each batch traverse process each handle each walk through list items",
        "records files documents rows urls messages pages entries",
    ],
    DifyNodeType.if_else: [
        "check judge decide route branch verify inspect compare evaluate condition status threshold",
        "risk level language intent priority permission quota quality score",
    ],
}


@dataclass(frozen=True)
class SemanticCandidate:
    node_type: DifyNodeType
    score: float
    sources: tuple[str, ...]


@dataclass(frozen=True)
class _SemanticDocument:
    node_type: DifyNodeType
    label: str
    text: str
    tokens: tuple[str, ...]


class SemanticBackend(Protocol):
    backend_name: str

    def search(self, text: str, top_k: int = 5) -> list[SemanticCandidate]:
        ...


class EmbeddingProvider(Protocol):
    provider_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class JsonEmbeddingCache:
    def __init__(self, cache_path: str) -> None:
        self._path = Path(cache_path)
        self._cache = self._load()

    def get(self, key: str) -> list[float] | None:
        value = self._cache.get(key)
        if not isinstance(value, list):
            return None
        return [float(item) for item in value]

    def set(self, key: str, vector: list[float]) -> None:
        self._cache[key] = vector
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._cache, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> dict[str, list[float]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): [float(item) for item in value]
            for key, value in data.items()
            if isinstance(value, list)
        }


class LocalHashEmbeddingProvider:
    provider_name = "local-hash"

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = _tokenize(text)
        if not tokens:
            return vector
        counts = Counter(tokens)
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * float(count)
        return _normalize_vector(vector)


class OpenAICompatibleEmbeddingProvider:
    provider_name = "openai-compatible"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._settings.semantic_embedding_api_key.strip():
            raise RuntimeError("SEMANTIC_EMBEDDING_API_KEY is not configured")

        url = f"{self._settings.semantic_embedding_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._settings.semantic_embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._settings.semantic_embedding_model,
            "input": texts,
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        rows = data.get("data", [])
        if not isinstance(rows, list):
            raise RuntimeError("Embedding response missing data array")
        embeddings: list[list[float]] = []
        for row in rows:
            embedding = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(embedding, list):
                raise RuntimeError("Embedding row missing embedding vector")
            embeddings.append([float(item) for item in embedding])
        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding response size does not match input size")
        return [_normalize_vector(vector) for vector in embeddings]


class TfidfSemanticBackend:
    backend_name = "tfidf"

    def __init__(
        self,
        prototypes: dict[DifyNodeType, list[str]] | None = None,
        rule_chunk_size: int = 24,
    ) -> None:
        self._prototypes = prototypes or DEFAULT_SEMANTIC_PROTOTYPES
        self._rule_chunk_size = rule_chunk_size
        self._documents = self._build_documents()
        self._idf = self._build_idf()
        self._doc_vectors = [self._vectorize(doc.tokens) for doc in self._documents]
        self._doc_norms = [self._vector_norm(vector) for vector in self._doc_vectors]

    def search(self, text: str, top_k: int = 5) -> list[SemanticCandidate]:
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        query_vector = self._vectorize(query_tokens)
        query_norm = self._vector_norm(query_vector)
        if query_norm == 0:
            return []

        raw_hits: list[tuple[int, float]] = []
        for index, doc_vector in enumerate(self._doc_vectors):
            score = self._cosine(query_vector, query_norm, doc_vector, self._doc_norms[index])
            if score > 0:
                raw_hits.append((index, score))

        return self._group_hits(raw_hits, top_k)

    def _group_hits(self, raw_hits: list[tuple[int, float]], top_k: int) -> list[SemanticCandidate]:
        raw_hits.sort(key=lambda item: item[1], reverse=True)
        if not raw_hits:
            return []

        grouped_scores: dict[DifyNodeType, list[tuple[float, str]]] = defaultdict(list)
        for index, score in raw_hits[: top_k * 4]:
            doc = self._documents[index]
            grouped_scores[doc.node_type].append((score, doc.label))

        candidates: list[SemanticCandidate] = []
        for node_type, hits in grouped_scores.items():
            hits.sort(key=lambda item: item[0], reverse=True)
            weighted_score = 0.0
            for rank, (score, _) in enumerate(hits[:3]):
                weighted_score += score * (0.85**rank)
            sources = tuple(label for _, label in hits[:2])
            candidates.append(
                SemanticCandidate(
                    node_type=node_type,
                    score=weighted_score,
                    sources=sources,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def _build_documents(self) -> list[_SemanticDocument]:
        documents: list[_SemanticDocument] = []
        for node_type, phrases in self._prototypes.items():
            for idx, phrase in enumerate(phrases):
                tokens = tuple(_tokenize(phrase))
                if tokens:
                    documents.append(
                        _SemanticDocument(
                            node_type=node_type,
                            label=f"prototype:{node_type.value}:{idx}",
                            text=phrase,
                            tokens=tokens,
                        )
                    )

            keywords = [
                keyword.replace("_", " ").replace("-", " ")
                for keyword in MAPPING_RULES_BY_NODE_TYPE.get(node_type.value, [])
                if _tokenize(keyword)
            ]
            for chunk_index in range(0, len(keywords), self._rule_chunk_size):
                chunk = keywords[chunk_index : chunk_index + self._rule_chunk_size]
                chunk_text = " ".join(chunk)
                tokens = tuple(_tokenize(chunk_text))
                if tokens:
                    documents.append(
                        _SemanticDocument(
                            node_type=node_type,
                            label=f"rules:{node_type.value}:{chunk_index // self._rule_chunk_size}",
                            text=chunk_text,
                            tokens=tokens,
                        )
                    )
        return documents

    def _build_idf(self) -> dict[str, float]:
        total_docs = len(self._documents)
        doc_freq: Counter[str] = Counter()
        for doc in self._documents:
            doc_freq.update(set(doc.tokens))
        return {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }

    def _vectorize(self, tokens: list[str] | tuple[str, ...]) -> dict[str, float]:
        token_counts = Counter(tokens)
        total = sum(token_counts.values())
        if total == 0:
            return {}
        return {
            token: (count / total) * self._idf.get(token, 1.0)
            for token, count in token_counts.items()
        }

    def _vector_norm(self, vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def _cosine(
        self,
        query_vector: dict[str, float],
        query_norm: float,
        doc_vector: dict[str, float],
        doc_norm: float,
    ) -> float:
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        common_tokens = set(query_vector) & set(doc_vector)
        dot = sum(query_vector[token] * doc_vector[token] for token in common_tokens)
        return dot / (query_norm * doc_norm)


class HybridSemanticBackend:
    backend_name = "hybrid"

    def __init__(
        self,
        lexical_backend: TfidfSemanticBackend | None = None,
        dense_backend: SemanticBackend | None = None,
        lexical_weight: float = 0.45,
        dense_weight: float = 0.55,
    ) -> None:
        self._lexical_backend = lexical_backend or TfidfSemanticBackend()
        self._dense_backend = dense_backend or self._lexical_backend
        self._lexical_weight = lexical_weight
        self._dense_weight = dense_weight

    def search(self, text: str, top_k: int = 5) -> list[SemanticCandidate]:
        lexical_hits = self._lexical_backend.search(text, top_k=top_k)
        dense_hits = self._dense_backend.search(text, top_k=top_k)
        merged: dict[DifyNodeType, dict[str, object]] = {}

        for candidate in lexical_hits:
            merged.setdefault(
                candidate.node_type,
                {"score": 0.0, "sources": []},
            )
            merged[candidate.node_type]["score"] = (
                float(merged[candidate.node_type]["score"]) + self._lexical_weight * candidate.score
            )
            merged[candidate.node_type]["sources"].extend(candidate.sources)

        for candidate in dense_hits:
            merged.setdefault(
                candidate.node_type,
                {"score": 0.0, "sources": []},
            )
            merged[candidate.node_type]["score"] = (
                float(merged[candidate.node_type]["score"]) + self._dense_weight * candidate.score
            )
            merged[candidate.node_type]["sources"].extend(candidate.sources)

        results = [
            SemanticCandidate(
                node_type=node_type,
                score=float(payload["score"]),
                sources=tuple(dict.fromkeys(payload["sources"]))[:3],
            )
            for node_type, payload in merged.items()
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


class RemoteEmbeddingBackend:
    backend_name = "remote-embedding"

    def __init__(
        self,
        settings: Settings | None = None,
        provider: EmbeddingProvider | None = None,
        fallback_backend: TfidfSemanticBackend | None = None,
        cache: JsonEmbeddingCache | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._fallback = fallback_backend or TfidfSemanticBackend()
        self._provider = provider or _build_embedding_provider(self.settings)
        self._cache = cache or JsonEmbeddingCache(self.settings.semantic_embedding_cache_path)
        self._documents = self._fallback._documents
        self._embedding_ready = True
        try:
            self._doc_vectors = [self._get_or_create_embedding(doc.text) for doc in self._documents]
            self._doc_norms = [_vector_norm(vector) for vector in self._doc_vectors]
        except Exception:
            self._embedding_ready = False
            self._doc_vectors = []
            self._doc_norms = []
        self._remote_enabled = bool(self.settings.semantic_embedding_api_key.strip())

    def search(self, text: str, top_k: int = 5) -> list[SemanticCandidate]:
        if not self._remote_enabled or not self._embedding_ready:
            return self._fallback.search(text, top_k=top_k)

        try:
            query_vector = self._get_or_create_embedding(text)
            query_norm = _vector_norm(query_vector)
            if query_norm == 0:
                return self._fallback.search(text, top_k=top_k)

            raw_hits: list[tuple[int, float]] = []
            for index, doc_vector in enumerate(self._doc_vectors):
                score = _cosine(query_vector, query_norm, doc_vector, self._doc_norms[index])
                if score > 0:
                    raw_hits.append((index, score))

            if not raw_hits:
                return self._fallback.search(text, top_k=top_k)
            return self._fallback._group_hits(raw_hits, top_k)
        except Exception:
            return self._fallback.search(text, top_k=top_k)

    def _get_or_create_embedding(self, text: str) -> list[float]:
        cache_key = self._cache_key(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        vector = self._provider.embed([text])[0]
        self._cache.set(cache_key, vector)
        return vector

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(
            f"{self._provider.provider_name}:{self.settings.semantic_embedding_model}:{text}".encode(
                "utf-8"
            )
        ).hexdigest()
        return digest


def build_semantic_backend(settings: Settings | None = None) -> SemanticBackend:
    resolved = settings or load_settings()
    backend_name = resolved.semantic_backend.strip().lower()
    lexical_backend = TfidfSemanticBackend()
    if backend_name in {"tfidf", ""}:
        return lexical_backend
    if backend_name in {"embedding", "remote-embedding"}:
        return RemoteEmbeddingBackend(settings=resolved, fallback_backend=lexical_backend)
    if backend_name in {"hybrid", "fusion"}:
        dense_backend = RemoteEmbeddingBackend(settings=resolved, fallback_backend=lexical_backend)
        return HybridSemanticBackend(
            lexical_backend=lexical_backend,
            dense_backend=dense_backend,
        )
    return lexical_backend


def _build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider_name = settings.semantic_embedding_provider.strip().lower()
    if provider_name in {"openai-compatible", "openai", ""}:
        return OpenAICompatibleEmbeddingProvider(settings)
    if provider_name in {"local-hash", "local"}:
        return LocalHashEmbeddingProvider()
    return LocalHashEmbeddingProvider()


def _tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("-", " ").replace("_", " ")
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = _vector_norm(vector)
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(query_vector: list[float], query_norm: float, doc_vector: list[float], doc_norm: float) -> float:
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    dot = sum(left * right for left, right in zip(query_vector, doc_vector))
    return dot / (query_norm * doc_norm)
