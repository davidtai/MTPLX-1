from __future__ import annotations


def test_public_reexports_support_and_settings_handlers():
    from mtplx.commands.public import (
        cmd_debug_public,
        cmd_doctor,
        cmd_settings_public,
    )
    from mtplx.commands.settings import cmd_settings_public as settings_impl
    from mtplx.commands.support import cmd_debug_public as debug_impl
    from mtplx.commands.support import cmd_doctor as doctor_impl

    assert cmd_settings_public is settings_impl
    assert cmd_debug_public is debug_impl
    assert cmd_doctor is doctor_impl


def test_public_reexports_model_and_integration_handlers():
    from mtplx.commands.integrations import cmd_integrate_public as integrate_impl
    from mtplx.commands.models import cmd_inspect_model_public as inspect_impl
    from mtplx.commands.models import cmd_model_public as model_impl
    from mtplx.commands.public import (
        cmd_inspect_model_public,
        cmd_integrate_public,
        cmd_model_public,
    )

    assert cmd_inspect_model_public is inspect_impl
    assert cmd_model_public is model_impl
    assert cmd_integrate_public is integrate_impl
