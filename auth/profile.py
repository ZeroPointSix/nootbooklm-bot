from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


class ProfileError(ValueError):
    pass


class ProfileManager:
    def __init__(self, path: str):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def install(
        self,
        state: dict[str, Any],
        *,
        verify: Callable[[Path], bool],
    ) -> None:
        self._validate(state)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=".storage-state-", suffix=".json"
        )
        temporary = Path(temporary_name)
        backup = self.path.with_suffix(".json.previous")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            if self.path.exists():
                os.replace(self.path, backup)
            os.replace(temporary, self.path)
            if not verify(self.path):
                raise ProfileError("NotebookLM 认证验证失败")
            if backup.exists():
                backup.unlink()
        except Exception:
            temporary.unlink(missing_ok=True)
            if backup.exists():
                self.path.unlink(missing_ok=True)
                os.replace(backup, self.path)
            elif self.path.exists():
                self.path.unlink()
            raise

    def logout(self) -> None:
        self.path.unlink(missing_ok=True)
        self.path.with_suffix(".json.previous").unlink(missing_ok=True)

    @staticmethod
    def _validate(state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ProfileError("Storage State 必须是对象")
        cookies = state.get("cookies")
        origins = state.get("origins")
        if not isinstance(cookies, list) or not isinstance(origins, list):
            raise ProfileError("Storage State 格式无效")
        if not any(
            isinstance(item, dict)
            and isinstance(item.get("domain"), str)
            and item["domain"].endswith(".google.com")
            for item in cookies
        ):
            raise ProfileError("Storage State 缺少 Google 会话")
