import json
from pathlib import Path
from typing import Iterator

from lxml import etree


class DataIterator:
    def iter_json(path: str) -> Iterator[dict]:
        """Iterate JSON data files provided by the wikiextractor package."""
        for file_path in Path(path).rglob("*"):
            if not file_path.is_file():
                continue
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        parsed_dict = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(e)
                        continue
                    if type(parsed_dict) is not dict:
                        continue
                    yield parsed_dict

    def iter_docs(path: str) -> Iterator[dict]:
        """Iterate Document data files provided by the wikiextractor package."""
        for file_path in Path(path).rglob("*"):
            if not file_path.is_file():
                continue
            with open(path, "rb") as fb:
                data = fb.read()
            wrapped = b"<root>" + data + b"</root>"
            parser = etree.XMLParser(recover=True, huge_tree=True)
            root = etree.fromstring(wrapped, parser=parser)
            for doc in root.findall(".//doc"):
                yield {
                    "id": doc.get("id"),
                    "url": doc.get("url"),
                    "title": doc.get("title"),
                    "category": doc.get("category"),
                    "text": doc.text.strip(),
                }
