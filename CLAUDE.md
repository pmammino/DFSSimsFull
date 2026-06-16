# Project conventions for Claude Code

## Branches & pull requests
- Do all work on a dedicated feature branch (never commit directly to `main`).
- **After pushing a batch of commits to a feature branch, open a pull request
  for it automatically** (base `main`) via the GitHub MCP tools — don't wait to
  be asked. If the branch already has an open PR, just push (the PR updates);
  only open a new PR when the previous one for that branch was merged/closed.
- Don't merge PRs; the maintainer merges from the GitHub UI.

## App
- `app.py` is the Streamlit front end (RotoWire full-dark theme, Tabbed
  Workspace: Setup · Players · Results · Export). See `README_app.md`.
- Brand fonts are served from `static/fonts/`; theme tokens in
  `.streamlit/config.toml`. Keep Windows-portable (the app runs on Windows).
