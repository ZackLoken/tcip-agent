"""Operator command sub-package: each module is one ``tcip`` subcommand's implementation.

The ``tcip`` console command's own dispatcher lives in ``tcip_web.cli``, the top of the stack,
whose install guarantees every command's imports; each module here exposes ``main(argv)``,
returning the exit code the dispatcher passes on.
"""
