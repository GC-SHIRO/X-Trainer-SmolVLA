from scripts.xtrainer.serve_policy import parse_args


def test_policy_server_accepts_reference_cli_aliases():
    args = parse_args(["--model-path", "checkpoint", "--use-length", "25"])

    assert args.checkpoint == "checkpoint"
    assert args.actions_per_chunk == 25


def test_policy_server_action_logging_is_disabled_by_default():
    args = parse_args([])

    assert args.log_actions is False
    assert args.action_log_path is None


def test_policy_server_accepts_action_log_options():
    args = parse_args(["--log-actions", "--action-log-path", "actions.jsonl"])

    assert args.log_actions is True
    assert args.action_log_path == "actions.jsonl"
