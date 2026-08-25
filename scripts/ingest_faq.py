"""
scripts/ingest_faq.py — Expand faq.json's structured KEC directory data into
granular Q&A pairs and bulk-load them into the RAG vector store.

faq.json has three kinds of source data:
  - "departments": per-department HOD + full faculty roster
  - "administration": college-level admin roles
  - "faq_style": a handful of hand-curated Q&A pairs already in FAQ form

Only "faq_style" is retrieval-ready as-is. This script turns the rest into
the same one-fact-per-page shape (HOD lookups, per-person designation
lookups, per-department roster lookups, per-admin-role lookups) so a query
about any single person/department/role retrieves one small, exact page
instead of relying on whatever the scraper happened to pull off the site.

Run from the project root: python -m scripts.ingest_faq
"""

import json

from rag import RAGService

FAQ_PATH = "data/faq.json"


def _dept_hod_qa(dept: dict) -> tuple[str, str]:
    name, hod = dept["department"], dept["hod"]
    email = dept.get("hod_email")
    answer = f"{hod} is the Head of the Department (HOD) of {name} at KEC."
    if email:
        answer += f" Contact: {email}."
    return f"Who is the HOD of the {name} at KEC?", answer


def _faculty_member_qa(person: dict, dept_name: str) -> tuple[str, str]:
    name, designation = person["name"], person["designation"]
    question = f"What is {name}'s designation at KEC?"
    answer = f"{name} is {designation} in the {dept_name} at KEC."
    return question, answer


def _dept_roster_qa(dept: dict) -> tuple[str, str]:
    name = dept["department"]
    lines = [f"{p['name']} ({p['designation']})" for p in dept["faculty"]]
    question = f"Who are the faculty members in the {name} at KEC?"
    answer = f"The faculty members in the {name} at KEC are: " + "; ".join(lines) + "."
    if dept.get("note"):
        answer += f" {dept['note']}"
    return question, answer


def _admin_qa(person: dict) -> tuple[str, str]:
    name, designation = person["name"], person["designation"]
    question = f"Who is the {designation} of KEC?"
    answer = f"{name} is the {designation} of Kathmandu Engineering College (KEC)."
    return question, answer


def build_pairs(data: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for dept in data.get("departments", []):
        pairs.append(_dept_hod_qa(dept))
        pairs.append(_dept_roster_qa(dept))
        for person in dept["faculty"]:
            pairs.append(_faculty_member_qa(person, dept["department"]))

    for person in data.get("administration", []):
        pairs.append(_admin_qa(person))

    for item in data.get("faq_style", []):
        pairs.append((item["question"], item["answer"]))

    # faq_style duplicates some auto-generated HOD questions (same wording,
    # richer answer) — keep the later (faq_style) version of any duplicate.
    deduped: dict[str, str] = {}
    for question, answer in pairs:
        deduped[question] = answer
    return list(deduped.items())


def main() -> None:
    with open(FAQ_PATH, encoding="utf-8") as f:
        data = json.load(f)

    pairs = build_pairs(data)
    print(f"Generated {len(pairs)} Q&A pairs from {FAQ_PATH}")

    rag = RAGService()
    results = rag.add_qa_batch(pairs)
    print(f"Ingested {len(results)} pages into the vector store")


if __name__ == "__main__":
    main()
