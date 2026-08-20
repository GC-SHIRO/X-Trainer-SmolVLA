"""Shared, reviewed X-trainer real-hardware defaults."""

from __future__ import annotations


# [left arm 6 joints, left gripper, right arm 6 joints, right gripper].
# This is the reset pose served to real-policy clients and must be reviewed for
# the installed tool, workspace, and joint limits before real-hardware use.
XTRAINER_RESET_POSE = (
    -1.57,
    0.0,
    -1.57,
    0.0,
    1.57,
    1.57,
    1.0,
    1.57,
    0.0,
    1.57,
    0.0,
    -1.57,
    -1.57,
    1.0,
)
