from pathlib import Path


def test_dockerignore_excludes_local_secrets_and_build_artifacts():
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    required_patterns = {
        ".env",
        ".env.*",
        "!.env.sample",
        ".notebooklm/",
        ".git",
        ".venv/",
        "logs/",
        "*.db",
    }

    missing = required_patterns.difference(dockerignore.splitlines())

    assert not missing
