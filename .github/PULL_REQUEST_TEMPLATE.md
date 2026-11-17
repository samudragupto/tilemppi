## Summary

Describe the changes in this pull request and link any related issues.

## Checklist

- [ ] `ruff format --check .` and `ruff check .` pass
- [ ] `mypy src/tilemppi` passes with zero errors
- [ ] `tilemppi verify-cutile` passes all `@ct.kernel` AST & parity checks
- [ ] `pytest --cov=tilemppi` passes (`>= 85%` coverage)
- [ ] Frontend `npm run lint && npm run typecheck && npm run test && npm run build` pass
- [ ] `python scripts/ci_check.py` exits `0`

