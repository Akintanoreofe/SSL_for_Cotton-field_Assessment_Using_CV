# Unreleased

These are the changes that will go out in the next release.

## Added

LeJEPA_Cotton_Field_Assessment started out as a project generated from [Cookiecutter PyPackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) containing:

- Initial scaffold for LeJEPA_Cotton_Field_Assessment.
- `src/lejepa_cotton_field_assessment/` package with CLI (Typer + Rich), py.typed marker
- Tests with pytest, coverage across Python 3.12/3.13/3.14
- CI via GitHub Actions: lint (Ruff), type check (ty), test matrix, coverage reporting
- Security scanning: CodeQL analysis for public repositories, Dependabot, and a Zizmor workflow audit
- Docs site with Zensical + mkdocstrings and a GitHub Pages deployment workflow
- Trusted publishing to PyPI with OIDC and build provenance attestation
- `justfile` with dev commands: qa, test, type-check, docs-serve, release
- Issue templates, PR template, contributing guide, code of conduct, security policy
- MIT license, .editorconfig, .gitignore

### Contributors

[@Akintanoreofe](https://github.com/Akintanoreofe) (Akintan Oreofeoluwa Anuoluwapo) created LeJEPA_Cotton_Field_Assessment.
