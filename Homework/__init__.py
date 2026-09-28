import os

# Each hwN_* package is self-contained (own routes, templates/, static/), so
# there's no shared static/template root here anymore - just the project
# root, kept in case a module needs to locate a sibling package's files.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
