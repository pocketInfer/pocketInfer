from __future__ import annotations

import json

import pytest

from pocketinfer.cli import main
from tests.config_samples import kimi_k3_config


def test_cli_writes_config_and_manifest(tmp_path, capsys) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "generated"
    source.write_text(json.dumps(kimi_k3_config()), encoding="utf-8")

    exit_code = main(
        [
            "scale",
            str(source),
            "--output-dir",
            str(output),
            "--memory-budget",
            "32GiB",
            "--kv-cache-budget",
            "6GiB",
            "--runtime-reserve",
            "6GiB",
        ]
    )

    assert exit_code == 0
    generated = json.loads((output / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "pocketinfer-manifest.json").read_text(encoding="utf-8")
    )
    report = (output / "fidelity-report.md").read_text(encoding="utf-8")
    assert generated["text_config"]["num_hidden_layers"] == 13
    assert manifest["adapter"] == "kimi-k3"
    assert "# PocketInfer fidelity report" in report
    assert "## Preserved" in report
    assert "wrote" in capsys.readouterr().out


def test_cli_refuses_to_replace_outputs_without_force(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "generated"
    source.write_text(json.dumps(kimi_k3_config()), encoding="utf-8")
    args = [
        "scale",
        str(source),
        "--output-dir",
        str(output),
        "--memory-budget",
        "32GiB",
        "--kv-cache-budget",
        "6GiB",
        "--runtime-reserve",
        "6GiB",
    ]

    assert main(args) == 0
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
