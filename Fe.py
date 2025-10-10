"""
EMIR Guidelines → Agentic Rules (Azure Document Intelligence → RAG + Multi-Agent) → YAML

- Scans scattered guidance across the doc using a BM25-ish retriever
- Iterates over your explicit field list
- Extracts actionable rules into 6 buckets + citations
- Writes a single YAML file (no third-party deps — includes a simple YAML dumper)

Customize:
- Plug your Azure OpenAI call in `LLMClient.call()`
- Provide your real DI object + full field list
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Iterable
import math, re, json
from collections import Counter

# =========================
# 1) LLM adapter (swap-in)
# =========================

class LLMClient:
    """
    Replace .call() with your Azure OpenAI chat completion.
    Must return TEXT. JSON will be parsed downstream.
    """
    def __init__(self, *, model: str = "gpt-4o", **config):
        self.model = model
        self.config = config

    def call(self, prompt: str, **kwargs) -> str:
        # TODO: implement Azure OpenAI call, e.g.:
        # from openai import AzureOpenAI
        # client = AzureOpenAI(api_key=..., api_version=..., azure_endpoint=...)
        # resp = client.chat.completions.create(
        #   model=self.model,
        #   messages=[{"role":"system","content":"You are a precise regulatory extraction agent. Return STRICT JSON only."},
        #             {"role":"user","content": prompt}],
        #   temperature=0.0, max_tokens=2000
        # )
        # return resp.choices[0].message.content
        #
        # Placeholder JSON scaffold to keep pipeline runnable:
        return json.dumps({
            "technical_validation": ["NO GUIDELINE"],
            "dependencies": [],
            "generation_rules": ["NO GUIDELINE"],
            "regulator_expectation": ["NO GUIDELINE"],
            "misc": [],
            "checklist": []
        })

# =========================
# 2) Document index (RAG)
# =========================

@dataclass
class DocChunk:
    page: int
    para_id: str
    text: str
    tokens: List[str] = field(default_factory=list)
    tf: Counter = field(default_factory=Counter)

class DocumentIndex:
    """
    Lightweight BM25-ish retrieval over Azure DI paragraphs.
    Keeps (page, para_id) for citations.
    """
    def __init__(self, di_obj: Dict[str, Any], tokenizer_regex: str = r"[A-Za-z0-9#\.\-]+", k1: float = 1.4, b: float = 0.75):
        self.rx = re.compile(tokenizer_regex)
        self.k1, self.b = k1, b
        self.chunks: List[DocChunk] = []
        self.df: Counter = Counter()
        self.N = 0
        self.avgdl = 0.0
        self._build(di_obj)

    def _iter_paragraphs(self, di_obj: Dict[str, Any]):
        for page in di_obj.get("pages", []):
            pnum = page.get("pageNumber")
            for p in page.get("paragraphs", []):
                pid = p.get("id") or f"p_{pnum}_{abs(hash(p.get('content',''))) % 10**8}"
                content = p.get("content", "")
                if isinstance(content, str) and content.strip():
                    yield (pnum, pid, content)

    def _tok(self, text: str) -> List[str]:
        return [t.lower() for t in self.rx.findall(text)]

    def _build(self, di_obj: Dict[str, Any]):
        dl_sum = 0
        for page, pid, txt in self._iter_paragraphs(di_obj):
            toks = self._tok(txt)
            tf = Counter(toks)
            self.chunks.append(DocChunk(page=page, para_id=pid, text=txt, tokens=toks, tf=tf))
            dl_sum += len(toks)
            for term in tf:
                self.df[term] += 1
        self.N = max(1, len(self.chunks))
        self.avgdl = dl_sum / self.N

    def _bm25(self, qtok: List[str], ch: DocChunk) -> float:
        score, dl = 0.0, max(1, len(ch.tokens))
        for t in qtok:
            f = ch.tf.get(t, 0)
            if not f: 
                continue
            df = self.df.get(t, 0)
            if df == 0: 
                continue
            idf = math.log(1 + (self.N - df + 0.5)/(df + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * ((f * (self.k1 + 1)) / denom)
        return score

    def search(self, query: str, top_k: int = 10) -> List[DocChunk]:
        qtok = self._tok(query)
        if not qtok: 
            return []
        scored = [(self._bm25(qtok, ch), ch) for ch in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k] if s > 0]

# =========================
# 3) Prompts
# =========================

EXTRACTION_SYSTEM = """You are a regulatory reporting expert extracting ACTIONABLE, MACHINE-CHECKABLE rules from the ESMA 'Guidelines on reporting under EMIR' (Oct 23, 2023).
Return STRICT JSON with keys:
- technical_validation: list[str]
- dependencies: list[{"field": str, "how": str, "impact": str}]
- generation_rules: list[str]
- regulator_expectation: list[str]
- misc: list[str]
- checklist: list[{"check": str, "severity": "REJECT"|"ERROR"|"WARN"|"INFO"}]
Use ONLY the provided passages; if absent, write "NO GUIDELINE" in that section.
"""

EXTRACTION_USER_TPL = """Field: {field_label}

Task:
From ONLY the passages, extract:
1) technical_validation (formats/enumerations/constraints),
2) dependencies (objects: field, how, impact),
3) generation_rules (scenarios/combos),
4) regulator_expectation (how field should look, if stated),
5) misc (extra critical info),
6) checklist (explicit checks with severities).

Passages (verbatim):
{passages}

IMPORTANT:
- No invented rules. If detail not present, put "NO GUIDELINE" in that section.
Return STRICT JSON only.
"""

DEPENDENCY_PROMPT = """You receive a JSON block for a field.
- Infer additional dependencies mentioned across sections and add them to 'dependencies' as:
  {"field":"<field label or number>", "how":"<dependency>", "impact":"<validation impact>"}
- If none, keep as-is.
Return STRICT JSON only.
"""

QA_PROMPT = """You are a QA agent. Ensure the JSON block has all list keys. 
If a list is empty and the passages didn’t contain info, add "NO GUIDELINE".
Deduplicate items.
Each checklist item must have a severity in {"REJECT","ERROR","WARN","INFO"} (default ERROR).
Return STRICT JSON only.
"""

# =========================
# 4) Agents
# =========================

@dataclass
class RetrievalResult:
    field_label: str
    passages: List[Dict[str, Any]]

class RetrievalAgent:
    def __init__(self, index: DocumentIndex):
        self.index = index

    def _query(self, field_label: str) -> str:
        return f"{field_label} validation format ISO MIC enumeration dependency generation regulator expectation checklist"

    def retrieve(self, field_label: str, k: int = 12) -> RetrievalResult:
        hits = self.index.search(self._query(field_label), top_k=k)
        passages = [{"page": h.page, "para_id": h.para_id, "text": h.text} for h in hits]
        return RetrievalResult(field_label, passages)

class ExtractionAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, field_label: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = []
        for p in passages:
            snippet = p["text"].strip()
            if len(snippet) > 1400:
                snippet = snippet[:1400] + " ..."
            merged.append(f"[page {p['page']} | {p['para_id']}] {snippet}")
        passages_text = "\n\n".join(merged) if merged else "(no passages found)"
        prompt = EXTRACTION_SYSTEM + "\n\n" + EXTRACTION_USER_TPL.format(field_label=field_label, passages=passages_text)
        try:
            return json.loads(self.llm.call(prompt))
        except Exception:
            return {
                "technical_validation": ["NO GUIDELINE"],
                "dependencies": [],
                "generation_rules": ["NO GUIDELINE"],
                "regulator_expectation": ["NO GUIDELINE"],
                "misc": [],
                "checklist": []
            }

class DependencyAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def refine(self, block: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(self.llm.call(DEPENDENCY_PROMPT + "\n\n" + json.dumps(block)))
        except Exception:
            return block

class QAAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def finalize(self, block: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(self.llm.call(QA_PROMPT + "\n\n" + json.dumps(block)))
        except Exception:
            # Minimal safety net
            out = {
                "technical_validation": block.get("technical_validation") or ["NO GUIDELINE"],
                "dependencies": block.get("dependencies") or [],
                "generation_rules": block.get("generation_rules") or ["NO GUIDELINE"],
                "regulator_expectation": block.get("regulator_expectation") or ["NO GUIDELINE"],
                "misc": block.get("misc") or [],
                "checklist": block.get("checklist") or []
            }
            for it in out["checklist"]:
                it.setdefault("severity", "ERROR")
            return out

# =========================
# 5) Orchestrator
# =========================

@dataclass
class FieldResult:
    field_label: str
    rules: Dict[str, Any]
    citations: List[Dict[str, Any]]

class Supervisor:
    def __init__(self, index: DocumentIndex, llm: LLMClient):
        self.retrieval = RetrievalAgent(index)
        self.extraction = ExtractionAgent(llm)
        self.dependency = DependencyAgent(llm)
        self.qa = QAAgent(llm)

    def process_field(self, field_label: str) -> FieldResult:
        ret = self.retrieval.retrieve(field_label)
        extracted = self.extraction.extract(field_label, ret.passages)
        with_deps = self.dependency.refine(extracted)
        final_rules = self.qa.finalize(with_deps)
        return FieldResult(
            field_label=field_label,
            rules=final_rules,
            citations=ret.passages
        )

    def run(self, field_list: List[str]) -> List[FieldResult]:
        return [self.process_field(f) for f in field_list]

# =========================
# 6) YAML writer (no deps)
# =========================

def _yaml_escape(s: str) -> str:
    if s is None:
        return "''"
    # Quote if contains special chars or leading/trailing spaces
    if re.search(r"[:#\-\?\[\]\{\},&*!|>'\"%@`]|^\s|\s$", s):
        # Use single quotes and escape existing single quotes
        return "'" + s.replace("'", "''") + "'"
    return s

def _yaml_dump(obj, indent=0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            key = _yaml_escape(str(k))
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.append(_yaml_dump(v, indent+1))
            else:
                val = _yaml_escape(str(v))
                lines.append(f"{sp}{key}: {val}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_yaml_dump(item, indent+1))
            else:
                lines.append(f"{sp}- {_yaml_escape(str(item))}")
        return "\n".join(lines)
    else:
        return f"{sp}{_yaml_escape(str(obj))}"

def write_rules_yaml(path: str, results: List[Dict[str, Any]]) -> None:
    """
    Writes a single YAML file with a top-level list where each item = field block.
    """
    # ensure field ordering within each block
    ordered = []
    for r in results:
        ordered.append({
            "field": r.get("field"),
            "technical_validation": r.get("technical_validation", []),
            "dependencies": r.get("dependencies", []),
            "generation_rules": r.get("generation_rules", []),
            "regulator_expectation": r.get("regulator_expectation", []),
            "misc": r.get("misc", []),
            "checklist": r.get("checklist", []),
            "citations": r.get("citations", [])
        })
    yaml_text = _yaml_dump(ordered)
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

# =========================
# 7) Public entry point
# =========================

def build_agentic_rules_yaml(
    di_obj: Dict[str, Any],
    field_list: List[str],
    yaml_path: str,
    llm_client: Optional[LLMClient] = None
) -> None:
    """
    Full pipeline:
    - Build index over Azure DI
    - For each field in field_list: Retrieval -> Extraction -> Dependency -> QA
    - Write single YAML file at yaml_path
    """
    llm = llm_client or LLMClient()
    index = DocumentIndex(di_obj)
    sup = Supervisor(index, llm)

    results = []
    for fr in sup.run(field_list):
        block = {
            "field": fr.field_label,
            **fr.rules,
            "citations": [
                {"page": c["page"], "para_id": c["para_id"], "snippet": (c["text"][:240] + ("..." if len(c["text"]) > 240 else ""))}
                for c in fr.citations
            ]
        ]
        results.append(block)

    write_rules_yaml(yaml_path, results)

# =========================
# 8) Example usage
# =========================

if __name__ == "__main__":
    # Example Azure DI object (replace with your real object)
    di_example = {
        "pages": [
            {
                "pageNumber": 1,
                "paragraphs": [
                    {"id": "p1_1", "content": "Reporting timestamp must be UTC with 'Z' timezone indicator."},
                    {"id": "p1_2", "content": "Venue of execution may be MIC or 'XXXX' depending on trading venue conditions."}
                ]
            },
            {
                "pageNumber": 2,
                "paragraphs": [
                    {"id": "p2_1", "content": "If MIC is used, ISIN could be required; otherwise UPI applies. Dependencies exist between venue and product identifiers."}
                ]
            }
        ]
    }

    # Your curated field list (iterate over exactly what you want to extract)
    field_names = [
        "1 ReportingTimestamp",
        "2 ReportSubmittingEntityID",
        "2.28 VenueOfExecution",
        "2.7 ISIN",
        "2.8 UPI",
    ]

    build_agentic_rules_yaml(
        di_obj=di_example,
        field_list=field_names,
        yaml_path="emir_guidelines_agentic_rules.yaml",
        llm_client=LLMClient(model="gpt-4o")  # replace with Azure OpenAI adapter
    )
    print("Wrote emir_guidelines_agentic_rules.yaml")
