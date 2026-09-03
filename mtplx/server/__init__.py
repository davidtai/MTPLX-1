"""Server entrypoints for MTPLX.

This package's ``__init__`` runs at exactly one useful moment: Python
executes it while resolving ``python -m mtplx.server.openai``, i.e. BEFORE
the first line of ``openai.py`` -- and therefore before that module's import
block pulls ``mtplx.runtime_options``, ``mtplx.generation``,
``mtplx.fable_block_verify`` and ``mtplx.fable_draft_k20_prescatter``, each
of which freezes several of the retained stack's env keys in a
module-level constant at ITS import.

The server's own retained-stack ``setdefault`` block does not run until
``mtplx/server/openai.py:_load``, thousands of imported lines later. For the
nine ``BIND_IMPORT`` keys that is too late: the environment would change and
no reader would ever look again, so the server would report lanes it had
not actually armed. :func:`~mtplx.full_stack_env.stamp_import_time_defaults` closes
that window and nothing else -- it stamps only that subset, only for a served
Flash-Next pack (or an explicit ``--profile turbo-full-stack``), never over an
operator's export or a disabled lane, and never raises.
"""

from __future__ import annotations

from ..full_stack_env import stamp_import_time_defaults as _stamp_early

__all__ = ["openai"]

#: What the early stamp actually put in place, for the receipt the server
#: prints once it has a stdout worth printing to. Empty on any other model
#: family, and empty when the operator had already exported all nine (or
#: disabled their lanes).
EARLY_STAMPED_ENV: dict[str, str] = _stamp_early()
