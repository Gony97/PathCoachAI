from openai import OpenAI
import json
import os, sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Tuple, Optional

# Get the path of the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (one level up from folder a)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)
from constants import API_KEY

@dataclass
class Module:
    id: str
    title: str
    description: str
    prerequisites: List[str]
    skills_required: List[str]
    estimated_time_hours: float
    resources: List[Dict[str, str]]
    status: str
    difficulty: str
    learning_style: List[str]
    outcome: str


def llm_client():

    return OpenAI(api_key=API_KEY)

MODULE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "prerequisites": {"type": "array", "items": {"type": "string"}},
        "skills_required": {"type": "array", "items": {"type": "string"}},
        "estimated_time_hours": {"type": "number"},
        "resources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "label": {"type": "string"},
                    "url": {"type": "string"}
                },
                "required": ["type", "label", "url"],
                "additionalProperties": False
            }
        },
        "status": {"type": "string", "enum": ["not started", "in progress", "completed"]},
        "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
        "learning_style": {"type": "array", "items": {"type": "string"}},
        "outcome": {"type": "string"}
    },
    "required": ["title","description","prerequisites","skills_required","estimated_time_hours",
                 "resources","status","difficulty","learning_style","outcome"],
    "additionalProperties": False
}

def generate_modules_with_llm(skills: list[str], base_minutes_if_beginner=120) -> list[dict]:
    """
    skills: list like ["python","sql", ...] already ranked; we’ll build one module per skill.
    Returns a list of JSON-serializable dicts matching MODULE_SCHEMA.
    """
    client = llm_client()
    model = "gpt-4o-mini"   # cost-efficient; change if you prefer another model
    results: list[dict] = []

    system_msg = (
        "You are a curriculum designer. For each input skill, return a single JSON object "
        "describing a concise learning module tailored for a data-science/algorithms job search. "
        "Keep resources practical (official docs or canonical guides). Use 1–3 resources."
    )

    for skill in skills:
        user_msg = (
            "Create a module for the given skill. Constraints:\n"
            f"- Target role: data scientist / algorithms engineer\n"
            f"- Skill: {skill}\n"
            "- Prerequisites should be short (IDs or SKILL: prefixes are fine).\n"
            "- estimated_time_hours: 1–6 (number only).\n"
            "- learning_style: pick from ['hands-on','reading','video','interactive','project']\n"
            "- status: 'not started'\n"
            "- outcome: start with an action verb.\n"
            "Return ONLY JSON."
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},  # JSON mode / structured output
        )
        obj = json.loads(resp.choices[0].message.content)

        # Light normalization / backstops
        try:
            obj = obj['module']
        except:
            continue
        obj.setdefault("status", "not started")
        
        if "prerequisites" not in obj or not obj["prerequisites"]:
            obj["prerequisites"] = [skill]
        if isinstance(obj.get("estimated_time_hours"), str):
            # if model returned "2h" etc., strip to a number
            obj["estimated_time_hours"] = float(re.sub(r"[^0-9.]", "", obj["estimated_time_hours"]) or 2.0)

        results.append(obj)

    return results


def make_module_id(i: int) -> str:
    return f"MOD-{i:03d}"


def save_json(mods: List[Module], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in mods], f, ensure_ascii=False, indent=2)


def save_markdown(mods: List[Module], path: str) -> None:
    headers = [
        "ID", "Name / Title", "Description", "Prerequisites", "Est. Time", "Resources", "Status", "Difficulty", "Learning Style", "Outcome"
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for m in mods:
        res = ", ".join([r.get("label", "link") for r in m.resources])
        prereq = ", ".join(m.prerequisites) if m.prerequisites else "—"
        ls = ", ".join(m.learning_style)
        lines.append(
            "| " + " | ".join([
                m.id,
                m.title,
                m.description,
                prereq,
                f"{m.estimated_time_hours:g}h",
                res,
                m.status,
                m.difficulty,
                ls,
                m.outcome,
            ]) + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":

    file_path = './code/generate_modules/modules_catalog.json'
    with open(file_path, 'r') as f:
        # Load the JSON data from the file into a Python object
        data = json.load(f)
    skills_list = []
    for skill_data in data:
        skills_list.append(skill_data['title'])

    print("Generating module content with the API…")
    llm_modules = generate_modules_with_llm(skills_list)

    # Convert LLM results into your Module dataclass objects (IDs, etc.)
    modules = []
    for i, m in enumerate(llm_modules, start=1):
        modules.append(Module(
            id=make_module_id(i),
                title=m["skill"],
            description=m["skill"],
            prerequisites=m.get("prerequisites", []),
            skills_required=m.get("prerequisites", []),
            estimated_time_hours=float(m.get("estimated_time_hours", 2.0)),
            resources=m.get("resources", []),
            status=m.get("status", "not started"),
            difficulty=m.get("difficulty", "beginner"),
            learning_style=m.get("learning_style", ["hands-on","reading"]),
            outcome=m.get("outcome")
        ))

    save_json(modules, "./code/generate_modules/updated_modules_catalog.json")
    save_markdown(modules, "./code/generate_modules/updated_modules_catalog.md")
    print("Wrote: modules_catalog.json, modules_catalog.md (with LLM-generated content)")
