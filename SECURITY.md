# Security

If you find a leaked credential, a way to game the scorer, or code that handles untrusted input unsafely, report it
through GitHub's private vulnerability reporting on the repository's Security tab. Please don't open a public issue.
We reply within a week.

Keep API keys in your environment or in a local `.env` file (see `.env.example`), never in a submission. CI runs
`scripts/check_secrets.py` on every pull request to catch key-like strings.

Dataset rebuilds download third-party files into `datasets/*/raw/`. Treat these files as untrusted. The builders open
them with `python -I` and never execute them.
