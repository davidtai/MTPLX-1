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
