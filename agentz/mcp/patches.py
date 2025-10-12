from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


PATCH_MARKER = "__browsermcp_original_close"
BUGGY_SNIPPET = "  server.close = async () => {\n    await server.close();"
PATCHED_SNIPPET = """  const __browsermcp_original_close = server.close.bind(server);
  server.close = async () => {
    if (server.__browsermcp_closing) {
      return;
    }
    server.__browsermcp_closing = true;
    await __browsermcp_original_close();
"""


def _candidate_paths(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        return []
    return base_dir.glob("**/node_modules/@browsermcp/mcp/dist/index.js")


def _prime_browsermcp_cache() -> bool:
    npx_path = shutil.which("npx")
    if not npx_path:
        logger.warning("Unable to locate `npx`; skipping Browser MCP patch.")
        return False

    try:
        subprocess.run(
            [
                npx_path,
                "--yes",
                "--package",
                "@browsermcp/mcp@latest",
                "node",
                "-e",
                "process.exit(0)",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("Failed to prime Browser MCP cache via npx: %s", exc)
        return False


def apply_browsermcp_close_patch() -> None:
    """Patch the Browser MCP CLI to avoid recursive `server.close` calls."""
    cache_root = Path.home() / ".npm" / "_npx"
    candidate_paths = list(_candidate_paths(cache_root))
    if not candidate_paths:
        if not _prime_browsermcp_cache():
            return
        candidate_paths = list(_candidate_paths(cache_root))

    for path in candidate_paths:
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s while applying Browser MCP patch: %s", path, exc)
            continue

        if PATCH_MARKER in original:
            continue

        if BUGGY_SNIPPET not in original:
            continue

        patched = original.replace(BUGGY_SNIPPET, PATCHED_SNIPPET, 1)
        if patched == original:
            continue

        try:
            path.write_text(patched, encoding="utf-8")
            logger.info("Patched Browser MCP close handler in %s", path)
        except OSError as exc:
            logger.warning("Failed to patch %s: %s", path, exc)
