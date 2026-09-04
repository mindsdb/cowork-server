"""Cowork-owned agentic coding domain.

This package is intentionally independent from the desktop's parked Claude
terminal prototype.  It exposes product concepts (sessions, turns, approvals,
events and workspaces); engine-specific wire objects stay inside adapters.
"""

from cowork.coding.service import CodingService, get_coding_service

__all__ = ["CodingService", "get_coding_service"]
