"""Local web UI for polyglot-lc.

This package drives the translator through the **CLI as a child process** rather
than importing :mod:`polyglot_lc`. That is a requirement, not a preference:
``polyglot_lc.config`` calls ``load_dotenv()`` at import time and
``load_dotenv`` never overrides a variable that is already set, so a server that
imported the library would push its own ``.env``-derived environment into every
child, and the child would then ignore ``.env`` entirely.

Only :mod:`webui.probe` imports the library, and only when it runs as a child
process of its own.
"""
