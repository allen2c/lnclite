"""Lightweight manager for named lnclite clients."""

import time
from dataclasses import dataclass

from lnclite.client import Lnclite
from lnclite.config import LncliteConfig
from lnclite.models import LncliteNotFoundError


@dataclass(frozen=True)
class ClientStats:
    name: str
    lancedb_path: str
    is_open: bool
    last_used_at: float | None


@dataclass(frozen=True)
class ManagerStats:
    max_clients: int
    idle_ttl_seconds: float | None
    active_clients: int
    clients: dict[str, ClientStats]


@dataclass
class _ManagedClient:
    config: LncliteConfig
    client: Lnclite | None = None
    last_used_at: float | None = None
    is_opening: bool = False


class LncliteManager:
    def __init__(
        self,
        *,
        configs: dict[str, LncliteConfig] | None = None,
        max_clients: int = 16,
        idle_ttl_seconds: float | None = 300,
    ) -> None:
        if max_clients < 1:
            raise ValueError(f"max_clients must be greater than 0, got {max_clients}")

        self.max_clients = max_clients
        self.idle_ttl_seconds = idle_ttl_seconds
        self._clients: dict[str, _ManagedClient] = {
            name: _ManagedClient(config=config)
            for name, config in (configs or {}).items()
        }

    async def open(
        self,
        name: str,
        *,
        config: LncliteConfig | None = None,
        create: bool = False,
    ) -> Lnclite:
        managed = self._clients.get(name)
        if managed is None:
            if config is None:
                raise LncliteNotFoundError(f"No config registered for {name}")
            managed = _ManagedClient(config=config)
            self._clients[name] = managed

        if config is not None:
            managed.config = config

        if managed.client is None:
            managed.is_opening = True
            try:
                managed.client = await self._open_client(managed.config, create=create)
            finally:
                managed.is_opening = False

        managed.last_used_at = time.time()
        await self._evict_if_needed(exclude=name)
        return managed.client

    async def close(self, name: str) -> None:
        managed = self._clients.get(name)
        if managed is None or managed.client is None:
            return
        await managed.client.close()
        managed.client = None

    async def close_all(self) -> None:
        for name in list(self._clients):
            await self.close(name)

    async def close_idle(self) -> None:
        if self.idle_ttl_seconds is None:
            return
        cutoff = time.time() - self.idle_ttl_seconds
        for name, managed in list(self._clients.items()):
            if managed.client is None or managed.is_opening:
                continue
            if managed.last_used_at is not None and managed.last_used_at < cutoff:
                await self.close(name)

    def stats(self) -> ManagerStats:
        clients = {
            name: ClientStats(
                name=name,
                lancedb_path=str(managed.config.lancedb_path),
                is_open=managed.client is not None,
                last_used_at=managed.last_used_at,
            )
            for name, managed in self._clients.items()
        }
        return ManagerStats(
            max_clients=self.max_clients,
            idle_ttl_seconds=self.idle_ttl_seconds,
            active_clients=sum(
                1 for client in self._clients.values() if client.client is not None
            ),
            clients=clients,
        )

    async def _open_client(self, config: LncliteConfig, *, create: bool) -> Lnclite:
        kwargs = {
            "lancedb_path": config.lancedb_path,
            "manifest_table": config.manifest_table,
            "document_table": config.document_table,
            "openai_embeddings_model": config.openai_embeddings_model,
            "model_settings": config.model_settings,
            "token_secret_key": config.token_secret_key,
            "vector_search_prefer": config.vector_search_prefer,
            "verbose": config.verbose,
            "index_cache_size": config.index_cache_size,
        }
        if create:
            return await Lnclite.new(**kwargs)
        return await Lnclite.load(**kwargs)

    async def _evict_if_needed(self, *, exclude: str) -> None:
        while self.stats().active_clients > self.max_clients:
            candidates = [
                (name, managed)
                for name, managed in self._clients.items()
                if name != exclude
                and managed.client is not None
                and not managed.is_opening
            ]
            if not candidates:
                return
            name, _ = min(candidates, key=lambda item: item[1].last_used_at or 0)
            await self.close(name)
