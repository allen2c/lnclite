"""SQL filter helpers for LanceDB queries."""

from typing import Literal

from lnclite.constants import ListOrder, SqlOrder


def quote_sql_string(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def tag_filter_any(tags: list[str]) -> str:
    values = ", ".join(quote_sql_string(tag) for tag in tags)
    return f"array_has_any(tags, [{values}])"


def tag_filter_all(tags: list[str]) -> str:
    values = ", ".join(quote_sql_string(tag) for tag in tags)
    return f"array_has_all(tags, [{values}])"


def tags_filter(
    *,
    tags_any: list[str] | None = None,
    tags_all: list[str] | None = None,
) -> str | None:
    filters: list[str] = []
    if tags_any:
        filters.append(tag_filter_any(tags_any))
    if tags_all:
        filters.append(tag_filter_all(tags_all))
    if not filters:
        return None
    return " AND ".join(f"({filter_})" for filter_ in filters)


def documents_list_where_clause(
    *,
    id_operator: Literal[">", "<"],
    sql_order: SqlOrder,
    after_id: int | None = None,
    tags_any: list[str] | None = None,
    tags_all: list[str] | None = None,
) -> str:
    id_filter = f"id {id_operator} {after_id}" if after_id is not None else "id > 0"
    filters = [id_filter]

    if filter_ := tags_filter(tags_any=tags_any, tags_all=tags_all):
        filters.append(filter_)

    where_clause = " AND ".join(f"({filter_})" for filter_ in filters)
    return f"{where_clause} ORDER BY id {sql_order}"


def to_sql_order(order: ListOrder) -> SqlOrder:
    if order in ("asc", 1):
        return "ASC"
    if order in ("desc", -1):
        return "DESC"
    raise ValueError(f"Invalid order: {order}")
