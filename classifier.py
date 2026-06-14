import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)

def load_labeled_examples() -> list[dict]:
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    label_definitions = """You are a podcast episode classifier. Classify episodes into exactly one of these four formats:

- interview: A host speaks with one or more guests in a conversational Q&A or discussion format. The guest's perspective is the main content.
- solo: A single host speaks from memory, personal experience, or opinion — no guests, no external interview clips as structural elements.
- panel: Multiple guests (3+) discuss a topic with roughly equal speaking time. Roundtables, debates, and group discussions.
- narrative: An episode assembled from external sources (interviews, archival audio, field recordings) into a story arc. The structure, not a single speaker's voice, carries the episode.

Here are labeled examples:
"""
    examples_text = ""
    for ex in labeled_examples:
        examples_text += f"\nDescription: {ex['description']}\nLabel: {ex['label']}\n"

    prompt = (
        label_definitions
        + examples_text
        + f"\nNow classify this new episode description:\nDescription: {description}\n\n"
        + "Respond with exactly this format:\nLabel: <one of: interview, solo, panel, narrative>\nReasoning: <one sentence explaining why>"
    )
    return prompt


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        response_text = response.choices[0].message.content.strip()

        label = "unknown"
        reasoning = response_text

        for line in response_text.splitlines():
            line_lower = line.lower().strip()
            if line_lower.startswith("label:"):
                raw_label = line.split(":", 1)[1].strip().lower().strip("*. ")
                if raw_label in VALID_LABELS:
                    label = raw_label
            elif line_lower.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error: {str(e)}"}
