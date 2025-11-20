# Contributing guidelines

Welcome dear chat-analyser contributor !

# Pull request guidelines

Commits must follow this syntax:

```bash
<type>(<impacted part>): <description> #<issue number>
```

with :

- type among: feat, enh, fix
- impacted part among: repo, docs, tests, core, api
- description: a one-sentence description of your changes 
- issue number: optional, the number of the issue(s) your PR adress

Note that your PR's commits will always be squashed to factorise related commits into a single one. If your PR contain multiples fixes, enhancements or features, it might be squashed into multiple commits before merging.

## Project set-up

### Dependency manager

This project uses [uv](https://docs.astral.sh/uv/) as a dependency manager. To install requirements run :

```bash
uv sync
```

To add a dependency simply run 

```bash
uv add <dependency>
```

To add a dependency for documentation, linting or testing purposes add it to the dev dependencies group with :

```bash
uv add <dependency> --dev
```

### Formatting and linting

This project rely on [ruff](https://docs.astral.sh/ruff/) to format and lint code base.

To format code simply run :

```bash
uv run ruff format .
```

For linting run

```bash
uv run ruff check
```

Some fixes might be fixable by using:

```bash
uv run ruff check --fix
```

### Testing

To run tests simply run :

```bash
uv run pytest .
```

### Building

To build your package's wheel simply run:

```bash
uv build
```
Your wheel file will be in `dist/`
