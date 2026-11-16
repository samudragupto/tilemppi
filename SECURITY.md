# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in **TileMPPI**, please report it responsibly:

1. **Do not** open a public GitHub issue for security-sensitive bugs.
2. Submit a private report via GitHub Security Advisories on the repository.
3. Include minimal reproduction steps, affected component (`server`, `protocol`, `cutile_compat`, or `frontend`), and impact assessment.

## Security Architecture Notes

- The binary `"TMPI"` WebSocket protocol strictly validates magic headers (`0x544D5049`), protocol versions, and buffer bounds before decoding payload slices.
- All REST API endpoints enforce strict `Pydantic v2` schema validation (`extra="forbid"`, bounded numeric ranges, power-of-two rollout constraints).
- The `cuTile` static AST verifier inspects `@ct.kernel` syntax trees purely via `ast.parse` without executing arbitrary untrusted strings.
