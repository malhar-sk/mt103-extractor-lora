# ADR 0001: Container runtime

- Status: accepted
- Date: 2026-11-24

## Context

The project runs as several services defined in `docker-compose.yml`. It is built on a shared 2 vCPU / 8 GB
Linux VPS, tested in GitHub Actions, and meant to be run by reviewers on Windows, macOS or Linux laptops.
Docker was the starting default, so the choice was re-examined instead of kept by habit. Options considered:
Docker (Desktop / Engine), Podman, OrbStack, and kind (Kubernetes in Docker).

## Decision

Keep **Docker** as the documented default, and treat the **Compose file as the contract**, not the tool.

- **Docker**: what most reviewers already have installed; GitHub Actions service containers are Docker-native;
  already installed on the build host. Docker Desktop is free for personal, education and small-business use.
- **Podman**: daemonless and rootless, which is a real security advantage on shared hosts, and it reads the
  same Compose file. Not the default because few reviewers have it installed and rootless networking makes
  some setups (e.g. WebRTC UDP ranges) harder.
- **OrbStack**: macOS only, so it cannot be the standard for Windows or Linux. Mac users who prefer it can run
  this project unchanged, because it runs standard Docker images and Compose files.
- **kind / Kubernetes**: adds a cluster to a project that needs a handful of containers on one machine.
  Not worth the complexity here.

## Podman compatibility

**Not tested yet.** The compose file uses only the standard Compose spec, so it should run under `podman-compose`, but that has not been verified for this repo. Ollama runs on CPU here, which should behave the same under Podman; GPU passthrough would need extra Podman setup.

## Consequences

- No Docker-only Compose extensions should be added, so other Compose-compatible tools keep working.
- TESTING.md documents Docker Desktop (Windows/macOS), Docker Engine (Linux) and Podman.
- Revisit if the project needs multi-host deployment (then Kubernetes), or if reviewers report they cannot use
  Docker Desktop (then make Podman a first-class, CI-tested path).
