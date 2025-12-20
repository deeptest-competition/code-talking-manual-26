import csv
import glob
import os

import pandas as pd
from lxml import etree

from model import Warning


def make_warning(row) -> Warning:
    warning_id = row["warning_id"]
    return Warning(
        id=warning_id,
        extra_ids=[],
        warning_text=row["warning_text"],
        top_section_id=warning_id.split(".SL")[0],
        top_section_title=row["top_section_title"],
    )


def read_warnings_from_csv(warnings_file) -> list[Warning]:
    df = pd.read_csv(warnings_file)
    df = df.dropna()
    text_to_warning: dict[str, Warning] = dict()
    for row in df.iterrows():
        warning = make_warning(row[1])
        if warning.warning_text in text_to_warning:
            text_to_warning[warning.warning_text].extra_ids.append(warning.id)
        else:
            text_to_warning[warning.warning_text] = warning
    return list(text_to_warning.values())


def extract_warnings_from_file(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    results = []

    for warning in root.findall(".//warning"):

        # ---- Background ID for the warning ----
        background = warning.xpath("ancestor::background[1]")
        warning_id = background[0].attrib.get("id", "") if background else ""

        # ---- Topic section (ancestor::section[2]) ----
        topic_section = warning.xpath("ancestor::section[2]")
        topic_section_id = topic_section_title = ""
        if topic_section:
            topic_section = topic_section[0]
            topic_section_id = topic_section.attrib.get("id", "")
            t = topic_section.find("title")
            if t is not None:
                topic_section_title = " / ".join(
                    pc.text.strip() for pc in t.findall("pcdata-case") if pc.text
                )
                if not topic_section_title and t.text:
                    topic_section_title = t.text.strip()

        # ---- Top-level (outermost) section ----
        top_section = warning.xpath("ancestor::section[last()]")
        top_section_id = top_section_title = ""
        if top_section:
            top_section = top_section[0]
            top_section_id = top_section.attrib.get("id", "")
            t2 = top_section.find("title")
            if t2 is not None:
                top_section_title = " / ".join(
                    pc.text.strip() for pc in t2.findall("pcdata-case") if pc.text
                )
                if not top_section_title and t2.text:
                    top_section_title = t2.text.strip()

        # ---- Fallback: use top section if topic section is empty ----
        if not topic_section_title:
            topic_section_title = top_section_title
            topic_section_id = top_section_id

        # ---- Warning text ----
        warning_text = " ".join(
            p.text.strip() for p in warning.findall(".//paragraph") if p.text
        )

        # ---- Detect type (first word: Danger, Warning, Notice) ----
        wtype = ""
        warning_text_lower = warning_text.lower()
        for label in ["danger", "warning", "notice"]:
            if warning_text_lower.startswith(label):
                wtype = label.capitalize()
                warning_text = warning_text[len(label) :].lstrip()
                break

        results.append(
            {
                "warning_id": warning_id,
                "top_section_id": top_section_id,
                "top_section_title": top_section_title,
                "topic_section_id": topic_section_id,
                "topic_section": topic_section_title,
                "type": wtype,
                "warning_text": warning_text,
            }
        )

    return results


def extract_warnings_from_all_files(data_folder, output_csv):
    all_files = glob.glob(os.path.join(data_folder, "*.xml"))
    merged_warnings = []

    for xml_file in all_files:
        print(f"Processing {xml_file}...")
        merged_warnings.extend(extract_warnings_from_file(xml_file))

    # Save merged warnings to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "warning_id",
                "top_section_id",
                "top_section_title",
                "topic_section_id",
                "topic_section",
                "type",
                "warning_text",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_warnings)

    print(
        f"Extracted {len(merged_warnings)} warnings from {len(all_files)} files and saved to {output_csv}"
    )


def get_warning_text(warnings_file, warning_id):
    # Load the CSV
    df = pd.read_csv(warnings_file)
    row = df[df["warning_id"] == warning_id]
    if not row.empty:
        return row.iloc[0]["warning_text"]
    return None
