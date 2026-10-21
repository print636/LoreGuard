<div align="center">

# LoreGuard

### Keep your story consistent. Keep the author in control.

An AI-assisted narrative consistency workspace for novelists and game narrative designers.

[简体中文](README.zh-CN.md) · **English**

[![Preview](https://img.shields.io/badge/status-preview-b7a2ff)](docs/current-stage-status.md) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-8be9f0)](LICENSE) [![CI](https://github.com/print636/LoreGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/print636/LoreGuard/actions/workflows/ci.yml)

[Explore the workspace](#inside-loreguard) · [Get started](#quick-start) · [Documentation](docs/README.md) · [Report an issue](https://github.com/print636/LoreGuard/issues)

</div>

![LoreGuard review workspace with story materials, selected chapters and review controls](docs/assets/readme/review-workspace.png)

<p align="center"><sub>Actual application UI with original demonstration data. The interface currently uses Simplified Chinese; these screenshots are not model-quality benchmarks.</sub></p>

## Your story grows. Its history should not get lost.

LoreGuard brings settings, character profiles, published chapters and new drafts into one review workflow. It helps you investigate potential continuity conflicts and follow the evidence back to the exact source version—without turning every uncertain model suggestion into a definitive error.

Use it while writing a chapter or when reviewing a batch of new story content. You can start with just the text; a complete world bible is not required.

## Inside LoreGuard

### One workspace for the material behind your story

| What you need | What LoreGuard provides |
| --- | --- |
| Bring existing writing together | Import Markdown, TXT, JSON and DOCX; organize documents and read their versions. |
| Review against the right context | Select new chapters and use author-confirmed settings and published history as context. Draft-only review is also available with limited coverage. |
| Find continuity conflicts | Check facts, locations, timelines, character knowledge, item ownership and world rules, using deterministic checks and optional model-assisted extraction. |
| Investigate, rather than guess | Read paired source evidence and open the frozen text used by the analysis. Formal issues, review clues and unverified proposals remain separate. |
| Check character consistency | Confirm character traits and compare new behavior against them. This is an **experimental, opt-in capability**, not a production OOC guarantee. |
| Keep the final decision | Record feedback, recheck revisions and export Markdown reports. Only the author can publish a draft into story history. |

### Read the evidence, not just a verdict

![LoreGuard report showing a formal issue, paired source evidence and author feedback](docs/assets/readme/evidence-report.png)

Potential conflicts come with explanations and source references. Review clues remain separate from formal issues, so missing context or a single unusual behavior does not silently become a definitive contradiction.

<details>
<summary>See how uncertain character behavior is kept as a review clue</summary>

![Separate review-clue section with character-setting and draft evidence](docs/assets/readme/review-clues.png)

This example needs author review; it is not counted or exported as a formal issue. The illustrated character workflow is experimental and opt-in.

</details>

<details>
<summary>More tools around the review workflow</summary>

- Document version comparison and analysis history.
- Optional relationship graph and timeline for completed runs.
- Persistent progress events, cancellation and controlled retries.
- Account-based projects and personal model configuration in authenticated deployments.
- In-app guidance and scoped draft recovery for supported editing areas.

These features have different operating boundaries; see the [user and developer reference](docs/developer-guide.en.md) before deployment.

</details>

## From new text to an author-reviewed report

1. **Import and organize.** Add settings, previous chapters and drafts; confirm which materials should count as authoritative context.
2. **Choose what to review.** Select the new chapter or batch. Configure your own model for AI-assisted analysis.
3. **Check the evidence.** Review issues and clues, record your decision, and recheck changes. Publish only when you choose to.

## Quick start

### Windows · local exploration

Install **Python 3.12** and **Node.js 22**, with **pnpm or Corepack** available.

```powershell
git clone https://github.com/print636/LoreGuard.git
cd LoreGuard
.\启动LoreGuard.cmd
```

The launcher installs dependencies, builds the web interface when needed and opens [http://127.0.0.1:8000](http://127.0.0.1:8000). Try the built-in original story example first.

### Docker Compose

With Docker and Compose installed, run from the repository root:

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080).

> No API key is needed for the deterministic examples. AI-assisted analysis requires model configuration; character checks, embeddings and tool-calling experiments are not all enabled by default. See [setup and capability switches](docs/developer-guide.en.md) and [account model settings](docs/account-model-provider.md). Local anonymous exploration is not a safe public deployment configuration.

## Documentation

| I want to… | Start here |
| --- | --- |
| Understand the product and review workflow | [Documentation hub](docs/README.md) · [In-app guide topics](docs/product-help-v1.md) |
| Configure models and optional capabilities | [Detailed reference](docs/developer-guide.en.md) · [Personal model credentials](docs/account-model-provider.md) |
| Understand the implementation | [Architecture](docs/architecture.md) · [Evidence retrieval](docs/evidence-retrieval-v1.md) |
| Deploy and operate an instance | [Production deployment](docs/production-deployment.md) · [Backup and restore](docs/backup-restore-rehearsal.md) |
| Check what is ready and what is not | [Current status](docs/current-stage-status.md) · [Human evaluation protocol](docs/character-ooc-closed-evaluation-v1.md) |
| Understand external references | [ConStory-Bench boundary](docs/constory-bench-boundary.md) |

Most detailed product and engineering documents are currently in Simplified Chinese. The long-form technical content formerly on this homepage is preserved in the [English](docs/developer-guide.en.md) and [Chinese](docs/developer-guide.zh-CN.md) references.

## Current boundaries

LoreGuard is a **Preview** tool that assists, rather than replaces, author review. Open-text quality still needs independent human evaluation. A completed run—or zero reported issues—does not mean every part of a story was understood or verified. Experimental character and tool-calling features require explicit configuration.

Use only text you own or are authorized to process. LoreGuard does not connect to or control commercial games.

## Contribute

Bug reports, reproducible examples and workflow suggestions are welcome in [Issues](https://github.com/print636/LoreGuard/issues). Please remove private manuscripts, credentials and personal information before sharing a report. For code changes, explain the intended behavior and include relevant tests.

## License

[Apache License 2.0](LICENSE).
