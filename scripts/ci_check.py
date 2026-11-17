"""Cross-platform full local CI verification runner for TileMPPI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd: list[str], cwd: Path | None = None, label: str = "") -> None:
    title = label or " ".join(cmd)
    print(f"\n=== [CI] {title} ===")
    proc = subprocess.run(cmd, cwd=cwd, check=False)  # nosec B603
    if proc.returncode != 0:
        print(f"[FAIL] {title} exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)
    print(f"[OK] {title}")


def _validate_workflows(repo_root: Path) -> None:
    print("\n=== [CI] Validating GitHub Actions YAML & lockfile ===")
    wf_dir = repo_root / ".github" / "workflows"
    workflows = sorted(wf_dir.glob("*.yml"))
    if not workflows:
        print("[FAIL] No workflow files found in .github/workflows/", file=sys.stderr)
        sys.exit(1)

    for wf in workflows:
        content = wf.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        if not isinstance(parsed, dict) or "jobs" not in parsed:
            print(f"[FAIL] Invalid GitHub Actions workflow structure in {wf}", file=sys.stderr)
            sys.exit(1)
        print(f"  [OK] {wf.name}: {len(parsed['jobs'])} job(s)")

    lockfile = repo_root / "frontend" / "package-lock.json"
    if not lockfile.is_file():
        print("[FAIL] frontend/package-lock.json is missing!", file=sys.stderr)
        sys.exit(1)
    print("  [OK] frontend/package-lock.json present")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    venv_ruff = repo_root / ".venv" / "Scripts" / "ruff.exe"
    if not venv_ruff.is_file():
        venv_ruff = repo_root / ".venv" / "bin" / "ruff"
    ruff_cmd = [str(venv_ruff)] if venv_ruff.is_file() else [py, "-m", "ruff"]

    _run([*ruff_cmd, "format", "--check", "."], cwd=repo_root, label="Ruff format check")
    _run([*ruff_cmd, "check", "."], cwd=repo_root, label="Ruff lint check")

    mypy_mod = subprocess.run(  # nosec B603
        [py, "-c", "import mypy"], capture_output=True, check=False
    )
    mypy_py = py if mypy_mod.returncode == 0 else (shutil.which("python") or py)
    _run([mypy_py, "-m", "mypy", "src/tilemppi"], cwd=repo_root, label="Mypy strict typecheck")

    bandit_check = subprocess.run(  # nosec B603
        [py, "-c", "import bandit"], capture_output=True, check=False
    )
    if bandit_check.returncode == 0:
        _run(
            [py, "-m", "bandit", "-c", "pyproject.toml", "-r", "src/tilemppi"],
            cwd=repo_root,
            label="Bandit security scan",
        )
    else:
        print("\n=== [CI] Bandit not installed locally; skipping local bandit scan ===")

    help_probe = subprocess.run(  # nosec B603
        [py, "-m", "pytest", "--help"], capture_output=True, text=True, check=False
    )
    pytest_cmd = [py, "-m", "pytest", "-o", "addopts=-ra -q"]
    if "--cov" in help_probe.stdout:
        pytest_cmd.extend(["--cov=tilemppi", "--cov-report=term-missing"])
    _run(pytest_cmd, cwd=repo_root, label="Pytest full suite")

    _run([py, "-m", "tilemppi.cli", "verify-cutile"], cwd=repo_root, label="cuTile AST & parity")
    _run(
        [py, "-m", "tilemppi.cli", "benchmark", "--quick"],
        cwd=repo_root,
        label="Quick benchmark suite",
    )

    npm_exe = shutil.which("npm")
    if npm_exe is None:
        print("[FAIL] npm not found on PATH", file=sys.stderr)
        sys.exit(1)

    fe_dir = repo_root / "frontend"
    _run([npm_exe, "run", "lint"], cwd=fe_dir, label="Frontend lint")
    _run([npm_exe, "run", "typecheck"], cwd=fe_dir, label="Frontend typecheck")
    _run([npm_exe, "run", "test"], cwd=fe_dir, label="Frontend tests")
    _run([npm_exe, "run", "build"], cwd=fe_dir, label="Frontend production build")

    _validate_workflows(repo_root)
    print("\n[OK] ALL TILEMPPI CI CHECKS PASSED (100% GREEN)!")


if __name__ == "__main__":
    main()
