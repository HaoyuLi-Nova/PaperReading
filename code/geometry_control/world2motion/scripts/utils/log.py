# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from typing import Any
import logging

_log = logging.getLogger("world2motion")


def info(message, *args, **kwargs):
    _log.info(message, *args)


def warning(message, *args, **kwargs):
    _log.warning(message, *args)


def debug(message, *args, **kwargs):
    _log.debug(message, *args)


def error(message, *args, **kwargs):
    _log.error(message, *args)


def critical(message, *args, **kwargs):
    _log.critical(message, *args)


def exception(message, *args, **kwargs):
    _log.exception(message, *args)
