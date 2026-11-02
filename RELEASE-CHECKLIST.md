# Before this repository goes public

We keep this repository private until the items below are done. They are decisions and data work, not code.

- [ ] Data licences. For Kleister-NDA, Kleister-Charity, VRDU, DocVQA, SROIE and the RealDoc PDFs
      ([LICENSE-DATA.md](LICENSE-DATA.md)), confirm the redistribution terms, ship only IDs and our labels and let
      `build.py` fetch the text, or drop the set.
- [ ] Decide the licence for our own additions. We propose CC BY 4.0.
- [ ] Give the Sage 2 row a publicly callable model ID and endpoint (today it is "rc9 preview"), and confirm with
      Levanto that we may publish the Sage 2 adapter (`clients.sage2_body`, `parse_sage2`).
- [ ] Decide who is named as author and maintainer in `CITATION.cff`, `pyproject.toml` and the README, and check
      those people for affiliation conflicts.
- [ ] Decide which "Not yet" items in [CONFLICTS.md](CONFLICTS.md) must be done before launch.
- [ ] Run `python scripts/check_secrets.py` and read the `notes` in every `results/*/metadata.json`.
- [ ] In the repository settings, enable private vulnerability reporting, protect `main`, and require the `tests`
      workflow.
