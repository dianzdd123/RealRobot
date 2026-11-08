# Contributing to RealRobot

Start with the [reuse guide](docs/REUSE.md), then read the code you intend to change. Issues are currently disabled on this fork. Describe a proposed fix in a pull request to this fork; do not assume its maintainers can change the upstream repository.

## Local workflow

1. Clone or fork this repository and create a focused branch.
2. Keep existing entry points, data formats, citations and third-party notices unless the change explicitly requires an update.
3. Make the smallest useful change. Update the README or reuse guide when setup, configuration, inputs or outputs change.
4. Run `python scripts/check_repository.py` and the relevant tests described in the reuse guide. Review `git diff --check` and the complete diff.
5. Open a pull request against `dianzdd123/RealRobot` with the problem, resulting behaviour and validation evidence.

## What makes a change reproducible

Include the commit, command, operating system, language/runtime and relevant dependency versions; expected versus observed behaviour; and a minimal public example. For research results, include the dataset/split, seed, checkpoint and evaluation configuration. Explain any checks that need unavailable data, hardware, GPU resources or external services.

Do not upload credentials, personal data, private course material, large model weights or raw datasets as part of a routine code fix. Keep examples small and use synthetic data where appropriate. Public reports should not contain private student work or assessed solutions.

## Validation boundaries

The structural check parses tracked Python and JSON/notebook files and checks local links in the README, reuse guide and this document. It does not import research modules, run notebook cells, execute archived code, start devices, validate remote URLs or reproduce experimental performance. Hardware scripts and diagnostic files may have side effects; read their entry points before running them.

## Attribution

This repository is a fork of [cb20600/RealRobot](https://github.com/cb20600/RealRobot). Original authorship, citations and upstream notices are retained. Changes here apply to this fork.

Existing licence terms remain unchanged. Where a repository-wide licence is absent, obtain clarification before redistributing or incorporating the code into another project. Keep licences for external models, datasets and assets separate from software provenance.
