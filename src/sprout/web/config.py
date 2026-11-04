"""Settings come from the environment; production refuses to start without a secret."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "knowledge" / "manifest.csv"


def _default_manifest() -> str:
    return str(DEFAULT_MANIFEST) if DEFAULT_MANIFEST.exists() else ""


MIN_SECRET_LENGTH = 16

# (allowed events, window in seconds)
DEFAULT_LIMITS: dict[str, tuple[int, int]] = {
    "login": (10, 60),
    "signup": (5, 3600),
    "upload": (20, 60),
    "chat": (30, 60),
}


@dataclass
class Config:
    ENV: str = "production"  # production | development | testing
    SECRET_KEY: str = ""
    DATA_DIR: str = "var/sprout"
    HTTPS: bool = False
    MANIFEST: str = field(default_factory=_default_manifest)  # gives chat sources their titles and links
    MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024
    API_TIMEOUT_SECONDS: float = 20.0
    # A generated answer is one search (seconds, mostly the CPU rerank) plus one model call. The model call gets 40 s
    # and the whole answer 50 s, which stays under gunicorn's default 60 s worker timeout. Chosen, not measured:
    # E8 records the generation time, and these two numbers should be revisited against its p95.
    # SPROUT_MEASUREMENT=1 (E8 only): lifts the rate limits, which would refuse a load test, and makes the upload and
    # chat routes report stage times in a Server-Timing header. /readyz says when it is on.
    MEASUREMENT: bool = False
    CHAT_TIMEOUT_SECONDS: float = 50.0
    CHAT_LLM_TIMEOUT_SECONDS: float = 40.0
    RATE_LIMITS: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(DEFAULT_LIMITS))

    @property
    def db_path(self) -> Path:
        return Path(self.DATA_DIR) / "sprout.db"

    @property
    def upload_dir(self) -> Path:
        return Path(self.DATA_DIR) / "uploads"

    def validate(self) -> None:
        if self.ENV not in ("production", "development", "testing"):
            raise RuntimeError(f"SPROUT_ENV must be production, development or testing, got {self.ENV!r}")
        if not self.SECRET_KEY.strip():
            raise RuntimeError("SPROUT_SECRET_KEY is required")
        if self.ENV != "testing" and len(self.SECRET_KEY.strip()) < MIN_SECRET_LENGTH:
            raise RuntimeError(f"SPROUT_SECRET_KEY must be at least {MIN_SECRET_LENGTH} characters")
        if not 0 < self.CHAT_LLM_TIMEOUT_SECONDS < self.CHAT_TIMEOUT_SECONDS:
            raise RuntimeError("the model call's timeout must be positive and shorter than the whole answer's")

    @classmethod
    def from_env(cls, env: dict[str, str]) -> Config:
        cfg = cls(ENV=env.get("SPROUT_ENV", "production"))
        cfg.SECRET_KEY = env.get("SPROUT_SECRET_KEY", "")
        cfg.DATA_DIR = env.get("SPROUT_DATA_DIR", cfg.DATA_DIR)
        cfg.MANIFEST = env.get("SPROUT_MANIFEST", cfg.MANIFEST)
        cfg.HTTPS = env.get("SPROUT_HTTPS", "") == "1"
        cfg.MAX_UPLOAD_BYTES = int(env.get("SPROUT_MAX_UPLOAD_BYTES", cfg.MAX_UPLOAD_BYTES))
        cfg.API_TIMEOUT_SECONDS = float(env.get("SPROUT_API_TIMEOUT_SECONDS", cfg.API_TIMEOUT_SECONDS))
        cfg.CHAT_TIMEOUT_SECONDS = float(env.get("SPROUT_CHAT_TIMEOUT_SECONDS", cfg.CHAT_TIMEOUT_SECONDS))
        cfg.CHAT_LLM_TIMEOUT_SECONDS = float(env.get("SPROUT_CHAT_LLM_TIMEOUT_SECONDS", cfg.CHAT_LLM_TIMEOUT_SECONDS))
        cfg.MEASUREMENT = env.get("SPROUT_MEASUREMENT", "") == "1"
        if cfg.MEASUREMENT:
            cfg.RATE_LIMITS = {k: (10**9, 60) for k in cfg.RATE_LIMITS}
        cfg.validate()
        return cfg
