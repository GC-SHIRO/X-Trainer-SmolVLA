from scripts.xtrainer.serve_policy import parse_args


def test_policy_server_accepts_reference_cli_aliases():
    args = parse_args(["--model-path", "checkpoint", "--use-length", "25"])

    assert args.checkpoint == "checkpoint"
    assert args.actions_per_chunk == 25
