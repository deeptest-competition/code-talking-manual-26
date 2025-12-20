from pathlib import Path

from lxml import etree


def parse_manual(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()

    chunks = []

    def traverse(node, parent_titles=[]):
        title = node.findtext("title")
        paragraphs = [p.text for p in node.findall("paragraph")]
        text = " ".join(filter(None, paragraphs))

        if title or text:
            chunks.append(
                {
                    "title": title,
                    "path": " > ".join(
                        parent_titles + [title] if title else parent_titles
                    ),
                    "content": text,
                    "id": node.get("id"),
                }
            )

        for child in node.findall("section"):
            traverse(child, parent_titles + [title] if title else parent_titles)

    traverse(root)
    return chunks


def load_chunks_from_directory(directory):
    manual_paths = list(Path(directory).glob("*.xml"))

    chunks = []
    for path in manual_paths:
        for c in parse_manual(path):
            c["manual_path"] = str(path)
            chunks.append(c)

    texts = [c["content"] for c in chunks]
    metadata = [
        {
            "title": c["title"],
            "path": c["path"],
            "id": c["id"],
            "manual_path": c["manual_path"],
        }
        for c in chunks
    ]
    return chunks, texts, metadata


def load_manuals_from_directory(directory):
    manual_paths = list(Path(directory).glob("*.xml"))
    manuals = []
    for path in manual_paths:
        with open(path, "r", encoding="utf-8") as f:
            manuals.append(f.read())
    return manuals
