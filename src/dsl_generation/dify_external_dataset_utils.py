from __future__ import annotations

import html
import hashlib
import http.client
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import yaml

from src.core.schema import DifyNodeType
from src.dsl_generation.eval_sample_utils import (
    ALLOWED_DIFY_NODE_TYPES,
    build_node_overlap_key,
    canonicalize_dsl_text,
)


REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_ISSUE_SEARCH_QUERIES = [
    'repo:langgenius/dify "kind: app" "workflow:"',
    'repo:langgenius/dify "DSL content"',
    'repo:langgenius/dify "```yaml" "kind: app"',
    'repo:langgenius/dify "type: if-else" "kind: app"',
    'repo:langgenius/dify "type: code" "kind: app"',
    'repo:langgenius/dify "type: tool" "provider_id:"',
    'repo:langgenius/dify "type: parameter-extractor"',
    'repo:langgenius/dify "type: variable-aggregator"',
]
DEFAULT_GIST_SEARCH_QUERIES = [
    'dify "kind: app" workflow',
    '"kind: app" "workflow:" dify yml',
    '"kind: app" "workflow:" dify yaml',
    '"kind: app" "type: parameter-extractor" dify',
    '"kind: app" "type: template-transform" dify',
    '"kind: app" "type: tool" "provider_id:" dify',
    '"kind: app" "type: http-request" dify',
    '"kind: app" "type: if-else" dify',
    '"kind: app" "type: iteration" dify',
    '"kind: app" "type: code" dify',
    '"kind: app" "type: llm" dify',
]
DEFAULT_REPO_SOURCES = [
    {
        "owner": "svcvit",
        "repo": "Awesome-Dify-Workflow",
        "branch": "main",
        "path_prefixes": ["DSL/"],
    },
    {
        "owner": "bingyue",
        "repo": "Dify-Workflow-DSL",
        "branch": "main",
        "path_prefixes": ["dsl/"],
    },
]


def request_text(url: str, *, sleep_seconds: float = 0.0) -> str:
    last_error: Exception | None = None
    for retry_index in range(3):
        try:
            request = urllib.request.Request(url, headers=REQUEST_HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8", errors="ignore")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return text
        except (HTTPError, URLError, TimeoutError, http.client.RemoteDisconnected) as exc:
            last_error = exc
            time.sleep(0.6 * (retry_index + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to request url: {url}")


def load_dataset_overlap_keys(dataset_path: Path) -> set[str]:
    overlap_keys: set[str] = set()
    if not dataset_path.exists():
        return overlap_keys

    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        try:
            dsl = yaml.safe_load(item["dsl"])
        except Exception:
            continue
        for node in dsl.get("workflow", {}).get("graph", {}).get("nodes", []):
            node_data = node.get("data", {})
            if node_data.get("type") not in ALLOWED_DIFY_NODE_TYPES:
                continue
            overlap_keys.add(build_node_overlap_key(node_data))
    return overlap_keys


def count_allowed_node_types(dsl_text: str) -> dict[str, int]:
    try:
        parsed = yaml.safe_load(dsl_text)
    except Exception:
        return {}

    counts: dict[str, int] = {}
    for node in parsed.get("workflow", {}).get("graph", {}).get("nodes", []):
        node_data = node.get("data", {})
        raw_type = node_data.get("type")
        if raw_type not in ALLOWED_DIFY_NODE_TYPES:
            continue
        mapped = ALLOWED_DIFY_NODE_TYPES[raw_type].value
        counts[mapped] = counts.get(mapped, 0) + 1
    return counts


def has_supported_nodes(dsl_text: str) -> bool:
    return any(count_allowed_node_types(dsl_text).values())


def extract_dsl_blocks_from_issue_html(issue_html: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(r'data-snippet-clipboard-copy-content=\\\"((?:[^\\\"]|\\.)*?)\\\"')
    for match in pattern.finditer(issue_html):
        escaped = match.group(1)
        try:
            decoded = json.loads(f"\"{escaped}\"")
        except json.JSONDecodeError:
            decoded = html.unescape(escaped)

        if "\\n" in decoded and "\n" not in decoded:
            decoded = decoded.replace("\\n", "\n")
        if "\\t" in decoded and "\t" not in decoded:
            decoded = decoded.replace("\\t", "\t")

        normalized = canonicalize_dsl_text(decoded)
        if "kind: app" in normalized and "workflow:" in normalized and "graph:" in normalized:
            blocks.append(normalized)
    return blocks


def collect_issue_ids(
    *,
    search_queries: list[str] | None = None,
    search_pages: list[int] | None = None,
) -> list[int]:
    issue_ids: set[int] = set()
    for query in search_queries or DEFAULT_ISSUE_SEARCH_QUERIES:
        for page in search_pages or [1, 2, 3]:
            encoded = urllib.parse.urlencode({"q": query, "type": "issues", "p": str(page)})
            search_url = f"https://github.com/search?{encoded}"
            try:
                html_text = request_text(search_url, sleep_seconds=0.15)
            except (HTTPError, URLError):
                continue
            for match in re.findall(r"/langgenius/dify/issues/([0-9]+)", html_text):
                issue_ids.add(int(match))
    return sorted(issue_ids)


def load_issue_page(issue_id: int, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"issue_{issue_id}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    issue_url = f"https://github.com/langgenius/dify/issues/{issue_id}"
    content = request_text(issue_url, sleep_seconds=0.15)
    cache_path.write_text(content, encoding="utf-8")
    return content


def collect_gist_urls(
    *,
    search_queries: list[str] | None = None,
    search_pages: list[int] | None = None,
    max_results: int = 120,
) -> list[str]:
    gist_urls: list[str] = []
    seen: set[str] = set()

    for query in search_queries or DEFAULT_GIST_SEARCH_QUERIES:
        for page in search_pages or [1]:
            encoded = urllib.parse.urlencode({"q": query, "p": str(page)})
            search_url = f"https://gist.github.com/search?{encoded}"
            try:
                html_text = request_text(search_url, sleep_seconds=0.15)
            except (HTTPError, URLError):
                continue

            for path in re.findall(r'href=\"(/[^/]+/[0-9a-f]{20,})\"', html_text):
                url = urllib.parse.urljoin("https://gist.github.com", path)
                if url in seen:
                    continue
                seen.add(url)
                gist_urls.append(url)
                if len(gist_urls) >= max_results:
                    return gist_urls
    return gist_urls


def load_gist_page(gist_url: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    gist_id = gist_url.rstrip("/").split("/")[-1]
    cache_path = cache_dir / f"gist_{gist_id}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    content = request_text(gist_url, sleep_seconds=0.15)
    cache_path.write_text(content, encoding="utf-8")
    return content


def extract_raw_urls_from_gist_html(gist_html: str) -> list[str]:
    raw_urls: list[str] = []
    for raw_path in re.findall(r'href=\"([^\"]+/raw/[^\"]+)\"', gist_html):
        raw_urls.append(urllib.parse.urljoin("https://gist.github.com", raw_path))
    return raw_urls


def load_raw_gist_file(raw_url: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{digest}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    content = request_text(raw_url, sleep_seconds=0.15)
    cache_path.write_text(content, encoding="utf-8")
    return content


def build_github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    quoted_path = urllib.parse.quote(path, safe="/")
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{quoted_path}"


def load_repo_tree(owner: str, repo: str, branch: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{owner}__{repo}__{branch}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    content = request_text(api_url, sleep_seconds=0.15)
    cache_path.write_text(content, encoding="utf-8")
    return json.loads(content)


def extract_repo_raw_file_urls_from_tree(
    tree_data: dict,
    *,
    owner: str,
    repo: str,
    branch: str,
    path_prefixes: list[str] | None = None,
) -> list[dict]:
    entries: list[dict] = []
    prefixes = tuple(path_prefixes or [])

    for item in tree_data.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = str(item.get("path", ""))
        if not path.lower().endswith((".yml", ".yaml")):
            continue
        if prefixes and not path.startswith(prefixes):
            continue

        entries.append(
            {
                "repo_id": f"{owner}/{repo}",
                "path": path,
                "source_url": build_github_raw_url(owner, repo, branch, path),
            }
        )
    return entries


def collect_repo_raw_file_entries(
    *,
    repo_sources: list[dict] | None = None,
    cache_dir: Path,
    max_results: int = 300,
) -> list[dict]:
    results: list[dict] = []
    seen_urls: set[str] = set()

    for source in repo_sources or DEFAULT_REPO_SOURCES:
        tree_data = load_repo_tree(
            source["owner"],
            source["repo"],
            source.get("branch", "main"),
            cache_dir,
        )
        entries = extract_repo_raw_file_urls_from_tree(
            tree_data,
            owner=source["owner"],
            repo=source["repo"],
            branch=source.get("branch", "main"),
            path_prefixes=source.get("path_prefixes"),
        )
        for entry in entries:
            if entry["source_url"] in seen_urls:
                continue
            seen_urls.add(entry["source_url"])
            results.append(entry)
            if len(results) >= max_results:
                return results
    return results


def load_raw_repo_file(raw_url: str, cache_dir: Path) -> str:
    return load_raw_gist_file(raw_url, cache_dir)


def maybe_parse_dify_dsl(raw_text: str) -> str | None:
    normalized = canonicalize_dsl_text(raw_text)
    if "kind: app" not in normalized or "workflow:" not in normalized or "graph:" not in normalized:
        return None
    if not has_supported_nodes(normalized):
        return None
    try:
        yaml.safe_load(normalized)
    except Exception:
        return None
    return normalized


def build_workflow_overlap_keys(dsl_text: str) -> set[str]:
    try:
        parsed = yaml.safe_load(dsl_text)
    except Exception:
        return set()

    overlap_keys: set[str] = set()
    for node in parsed.get("workflow", {}).get("graph", {}).get("nodes", []):
        node_data = node.get("data", {})
        if node_data.get("type") not in ALLOWED_DIFY_NODE_TYPES:
            continue
        overlap_keys.add(build_node_overlap_key(node_data))
    return overlap_keys


def build_source_metadata(
    *,
    source_type: str,
    source_url: str,
    dsl_text: str,
) -> dict:
    parsed = yaml.safe_load(dsl_text)
    graph = parsed.get("workflow", {}).get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_type_counts = count_allowed_node_types(dsl_text)
    return {
        "source": source_type,
        "source_url": source_url,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "supported_node_type_counts": node_type_counts,
    }
