import asyncio
import json
import logging
import os
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        min_size: int = 2,
        max_size: int = 5,
        **pool_kwargs: Any,
    ):
        self.host = host or os.getenv("PGHOST")
        self.port = port or int(os.getenv("PGPORT", "5432"))
        self.database = database or os.getenv("PGDATABASE")
        self.user = user or os.getenv("PGUSER")
        self.password = password or os.getenv("PGPASSWORD")

        self.min_size = int(os.getenv("PGPOOL_MIN_SIZE", str(min_size)))
        self.max_size = int(os.getenv("PGPOOL_MAX_SIZE", str(max_size)))

        if self.host and "database.azure.com" in self.host:
            pool_kwargs.setdefault("ssl", "require")

        self.pool_kwargs = pool_kwargs
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_loop: Optional[asyncio.AbstractEventLoop] = None

    async def _connection_init(self, conn: asyncpg.Connection) -> None:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
            format="text",
        )
        await conn.set_type_codec(
            "json",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
            format="text",
        )

    async def get_pool(self) -> asyncpg.Pool:
        current_loop = asyncio.get_running_loop()
        if self._pool is None or self._pool_loop is not current_loop:
            if self._pool:
                await self._pool.close()

            logger.info(
                f"Creating connection pool: {self.host}:{self.port}/{self.database} "
                f"(min_size={self.min_size}, max_size={self.max_size})"
            )

            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60,
                init=self._connection_init,
                **self.pool_kwargs,
            )
            self._pool_loop = current_loop
            logger.info("Connection pool created successfully")

        return self._pool

    async def execute(self, query: str, *args: Any) -> str:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def executemany(self, query: str, args: list[tuple]) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(query, args)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._pool_loop = None
            logger.info("Connection pool closed")

    async def test_connection(self) -> bool:
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False