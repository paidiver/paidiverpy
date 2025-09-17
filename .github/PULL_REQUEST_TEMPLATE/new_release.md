---
name: Prepare for next release
about: 'release'
title: 'Prepare for next release'
labels: 'release'
assignees: ''
---

# Setup

- [ ] Make sure you are on the `dev` branch
- [ ] Increase release version by using  ``bump-my-version bump major``, ``bump-my-version bump minor`` or ``bump-my-version bump patch`` command
- [ ] Create a PR to prepare it, name it with one of the [Nature emoji](https://www.webfx.com/tools/emoji-cheat-sheet/#tabs-3) and make sure it was [never used before](https://github.com/paidiver/paidiverpy/pulls?q=is%3Apr+label%3Arelease+)
# Prepare code for release

## Code clean-up
- [ ] Run [ruff](https://github.com/astral-sh/ruff) from repo root and fix errors: ``ruff check && ruff format --check``

## Software distribution readiness
- [ ] Manually trigger [upstream CI tests](https://github.com/paidiver/paidiverpy/actions/workflows/pytests-upstream.yml) for the release branch and ensure they are passed
- [ ] Make sure that all CI tests are passed
- [ ] [Activate](https://readthedocs.org/projects/paidiverpy/versions/) and make sure the documentation for the release branch is [built on RTD](https://readthedocs.org/projects/paidiverpy/builds/)

## Preparation conclusion
- [ ] Merge this PR to main
- [ ] Make sure all CI tests are passed and RTD doc is built on the main branch

# Publish the release

- [ ] ["Draft a new release"](https://github.com/paidiver/paidiverpy/releases/new) on GitHub.
Choose a release tag vX.Y.Z, fill in the release title and click on the `Auto-generate release notes` button.
This will trigger the [publish Github action](https://github.com/paidiver/paidiverpy/blob/main/.github/workflows/release.yml) that will push the release on [Pypi](https://pypi.org/project/paidiverpy/#history).
- [ ] Last check if the release version on the files is correct and that the [documentation is ready](https://readthedocs.org/projects/paidiverpy/builds/)
- [ ] Publish !
- [ ] Checkout on [Pypi](https://pypi.org/project/paidiverpy/#history) and [Conda](https://github.com/conda-forge/paidiverpy-feedstock/pulls) that the new release is distributed (Conda is not implemented yet)

[![Publish on pypi](https://github.com/paidiver/paidiverpy/actions/workflows/release.yml/badge.svg)](https://github.com/paidiver/paidiverpy/actions/workflows/release.yml)

# CI tests / RTD build results
[![CI tests Upstream](https://github.com/paidiver/paidiverpy/actions/workflows/pytests-upstream.yml/badge.svg?branch=main)](https://github.com/paidiver/paidiverpy/actions/workflows/pytests-upstream.yml)
[![Documentation Status](https://readthedocs.org/projects/paidiverpy/badge/?version=latest)](https://paidiverpy.readthedocs.io/en/latest)
