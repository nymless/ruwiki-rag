import json
from pathlib import Path
from typing import Iterator

from lxml import etree


class DataIterator:
    @staticmethod
    def iter_json(path: str) -> Iterator[dict]:
        """Iterate JSON lines files produced by wikiextractor (one JSON per line)."""
        for file_path in Path(path).rglob("*"):
            if not file_path.is_file():
                continue
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            yield obj
            except Exception as e:
                print(f"[iter_json] {file_path}: {e}")

    @staticmethod
    def iter_docs(path: str) -> Iterator[dict]:
        """Iterate <doc>...</doc> blocks from wikiextractor XML-like files."""
        for file_path in Path(path).rglob("*"):
            if not file_path.is_file():
                continue
            try:
                with file_path.open("rb") as fb:
                    data = fb.read()
                wrapped = b"<root>" + data + b"</root>"
                parser = etree.XMLParser(recover=True, huge_tree=True)
                root = etree.fromstring(wrapped, parser=parser)
                for doc in root.findall(".//doc"):
                    yield {
                        "id": doc.get("id") or "",
                        "url": doc.get("url") or "",
                        "title": doc.get("title") or "",
                        "category": doc.get("category") or "",
                        "text": (doc.text or "").strip(),
                    }
            except Exception as e:
                print(f"[iter_docs] {file_path}: {e}")
