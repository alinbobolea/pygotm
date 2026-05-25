"""Stdio JSON-RPC warm daemon for process-boundary integrations."""

from pygotm.serve.daemon import serve_forever, warm_kernel

__all__ = ["serve_forever", "warm_kernel"]
