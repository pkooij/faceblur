"""Vendored from LeRobot (https://github.com/huggingface/lerobot, src/lerobot/streaming, Apache-2.0, commit 17aea13f).

Only the MP4 sample-table reader (`mp4.py`) and the pooled HTTP byte-range fetcher (`range_fetch.py`) are used.
Local change: `range_fetch._resolve_url` accepts the 3xx redirect response that huggingface_hub 2.x raises on.
"""
