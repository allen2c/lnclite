import keyword
import random
import re
from pathlib import Path

import lines_of_work as low
from tqdm import tqdm

NUM_WORK = 100

TARGET_DIR = "data/works"


def to_valid_var(name: str) -> str:
    """Convert a string to a valid Python variable name.

    Examples:
    - "any string" -> "any_string"
    - "123-data!"  -> "_123_data_"
    - "class"      -> "class_"
    """
    # 1. Replace all non-word characters (anything not [a-zA-Z0-9_]) with underscore
    name_ = re.sub(r"\W+| ", "_", name)

    # 2. Ensure it doesn't start with a digit
    if name_[0].isdigit():
        name_ = "_" + name_

    # 3. Handle Python keywords
    if keyword.iskeyword(name_):
        name_ += "_"

    name_ = name_.lower().strip()
    if not name_:
        raise ValueError(f"Invalid name: {name}")
    return name_


def dump_work(work: low.Work, target_dir: str):
    target_dirpath = Path(target_dir)
    target_dirpath.mkdir(parents=True, exist_ok=True)

    work_dirpath = target_dirpath.joinpath(f"{work.work_id}")
    work_dirpath.mkdir(parents=True, exist_ok=True)

    work_dirpath.joinpath("agent.json").write_text(work.agent.model_dump_json())
    for knowledge in work.iter_all_knowledge():
        work_dirpath.joinpath(f"{to_valid_var(knowledge.title)}.md").write_text(
            f"# {knowledge.title}\n\n{knowledge.content}"
        )


def main(num_work: int, target_dir: str):
    works = list(low.iter_all_works())
    random.shuffle(works)
    works = works[:num_work]

    for work in tqdm(works):
        dump_work(work, target_dir)


if __name__ == "__main__":
    main(num_work=NUM_WORK, target_dir=TARGET_DIR)
