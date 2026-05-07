"""Search a small downloaded open dataset with lnclite."""

import asyncio
import csv
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import DocumentCreate, Lnclite, get_openai_embeddings_model

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
DATA_LICENSE = "CC0 public domain via the Palmer Penguins teaching dataset"
DATA_PATH = Path("outputs/examples/data/penguins.csv")
STORE_PATH = Path("outputs/examples/open-dataset-search.lance")
ROW_LIMIT = 40


async def main() -> None:
    if STORE_PATH.exists():
        shutil.rmtree(STORE_PATH)

    csv_path = download_dataset()
    documents = load_penguin_documents(csv_path, limit=ROW_LIMIT)

    embeddings = get_openai_embeddings_model(openai_client=AsyncOpenAI())
    client = await Lnclite.new(
        lancedb_path=STORE_PATH,
        openai_embeddings_model=embeddings,
        model_settings=ModelSettings(dimensions=1536),
        name="Open dataset search",
        description=f"Palmer Penguins demo data. Source: {DATA_URL}. {DATA_LICENSE}.",
    )

    try:
        await client.documents.batch_create(documents)
        await client.create_index()

        results = await client.search(
            "Which penguins have long flippers and high body mass?",
            limit=5,
        )
        print("Top penguin matches:")
        for index, result in enumerate(results.results, start=1):
            print(f"{index}. {result.document.content}")
            print(f"   tags: {', '.join(result.document.tags)}")
            print(f"   distance: {result.distance:.4f}")
    finally:
        await client.close()


def download_dataset() -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading {DATA_URL}")
        urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def load_penguin_documents(path: Path, *, limit: int) -> list[DocumentCreate]:
    documents: list[DocumentCreate] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if len(documents) >= limit:
                break
            if not _is_complete(row):
                continue
            documents.append(_document_from_row(row))
    return documents


def _document_from_row(row: dict[str, str]) -> DocumentCreate:
    species = row["species"]
    island = row["island"]
    sex = row["sex"].lower()
    bill_length = row["bill_length_mm"]
    bill_depth = row["bill_depth_mm"]
    flipper_length = row["flipper_length_mm"]
    body_mass = row["body_mass_g"]
    content = (
        f"{species} penguin observed on {island} island. "
        f"The {sex} penguin has a bill length of {bill_length} mm, "
        f"a bill depth of {bill_depth} mm, a flipper length of "
        f"{flipper_length} mm, and a body mass of {body_mass} g."
    )
    return DocumentCreate(
        content=content,
        tags=[
            "dataset:palmer-penguins",
            f"species:{species.lower()}",
            f"island:{island.lower()}",
            f"sex:{sex}",
        ],
    )


def _is_complete(row: dict[str, str]) -> bool:
    required_columns = [
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]
    return all(row.get(column) for column in required_columns)


if __name__ == "__main__":
    asyncio.run(main())
