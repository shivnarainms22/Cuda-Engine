from importlib.resources import files


def load_prompt(name: str) -> str:
    prompt_path = files(__package__).joinpath(f"{name}.md")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt not found: {name}")
    return prompt_path.read_text(encoding="utf-8")
