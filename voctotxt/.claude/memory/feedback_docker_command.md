---
name: Docker compose command preference
description: User prefers full docker compose command with explicit -f flag
type: feedback
---

Always use `docker compose -f docker-compose.local.yml` for all Docker operations in this project.

**Why:** User explicitly prefers the full command over `just` shortcuts or bare `docker compose`.

**How to apply:** Every docker compose invocation should be `docker compose -f docker-compose.local.yml <subcommand>`.
