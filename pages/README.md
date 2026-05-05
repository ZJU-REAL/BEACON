# BEACON Project Page

Static project page for **BEACON: Milestone-Guided Policy Learning for Long-Horizon Language Agents**.

Deployed automatically to GitHub Pages from `master` via [`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml) whenever files under `pages/` change.

## Local preview

```bash
cd pages
python3 run_server.py            # opens http://localhost:8000
python3 run_server.py 8080       # custom port
```

Or with the built-in server:

```bash
python3 -m http.server 8000 --directory pages
```

## Layout

```
pages/
├── index.html              # main page
├── .nojekyll               # disable Jekyll on GitHub Pages
├── run_server.py           # local preview helper
└── static/
    ├── css/index.css       # custom styles (Bulma loaded via CDN)
    ├── js/                 # (reserved for future scripts)
    └── images/             # figures shown on the page
```
