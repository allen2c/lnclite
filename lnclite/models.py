"""Public models for lnclite."""

from pydantic import BaseModel, Field, model_validator


class DocumentCreate(BaseModel):
    content: str
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_values(self) -> "DocumentCreate":
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("Content cannot be empty")
        return self


class Document(BaseModel):
    id: int
    content: str
    md5: str
    vector: list[float] | None = None
    tags: list[str]


class ManifestModel(BaseModel):
    id: int
    name: str
    description: str
    model: str
    dimensions: int
    last_updated: int


class SearchResult(BaseModel):
    document: Document
    distance: float


class SearchResults(BaseModel):
    results: list[SearchResult]


class LncliteNotFoundError(Exception):
    """Raised when a requested lnclite resource does not exist."""
