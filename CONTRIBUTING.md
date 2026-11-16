# Contributing to TileMPPI

Thank you for contributing to **TileMPPI**!

## Development Setup

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e ".[server,dev]"
```

For the 3D frontend playground:

```bash
cd frontend
npm ci
npm run test
npm run build
```

## Coding Standards & Verification

1. **Tile Kernel Discipline:** All `@ct.kernel` functions under `src/tilemppi/kernels/cutile/` must pass `tilemppi verify-cutile` with zero static AST contract violations and exact numerical parity against `reference_cpu`.
2. **Type Safety & Linting:**
   ```bash
   ruff format --check .
   ruff check .
   mypy src/tilemppi
   bandit -c pyproject.toml -r src/tilemppi
   ```
3. **Full CI Parity Check:**
   ```bash
   python scripts/ci_check.py
   ```
4. **Commit Messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `ci:`, `chore:`).
