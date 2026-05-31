"""First-party channel plugins.

Each plugin is a module here exposing a module-level ``plugin: ChannelPlugin``
(e.g. ``telegram.py``). :func:`cowork.channels.registry.load_first_party_plugins`
imports every module in this package and registers the ones that declare a
``plugin``. Empty for now — the first plugin (Telegram) lands in a later slice.
"""
from __future__ import annotations
