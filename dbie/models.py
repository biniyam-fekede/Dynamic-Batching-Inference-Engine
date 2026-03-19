"""Core data structures for DBIE."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Request:
    """A single inference request — the fundamental unit flowing through the system."""

    payload: torch.Tensor
    future: asyncio.Future
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    arrival_time: float = field(default_factory=time.monotonic)


@dataclass
class Batch:
    """A batched group of requests ready for inference."""

    requests: List[Request]
    tensor: torch.Tensor
    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    creation_time: float = field(default_factory=time.monotonic)

    @classmethod
    def from_requests(cls, requests: List[Request]) -> Batch:
        """Build a Batch by stacking request payloads.

        Raises ValueError (and resolves all futures with the exception)
        if tensor shapes are incompatible.
        """
        try:
            tensor = torch.stack([r.payload for r in requests])
        except RuntimeError as exc:
            err = ValueError(f"Tensor shape mismatch in batch: {exc}")
            for r in requests:
                if not r.future.done():
                    r.future.set_exception(err)
            raise err
        return cls(requests=requests, tensor=tensor)

    @property
    def size(self) -> int:
        return len(self.requests)
