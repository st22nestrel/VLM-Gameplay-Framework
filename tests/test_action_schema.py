"""Action config decoding must be independent of Windows' default code page."""

import pytest

from src.actions import schema


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_load_utf8_actions_with_non_utf8_system_default(tmp_path, monkeypatch, encoding):
    path = tmp_path / "actions.yaml"
    path.write_text(
        "actions:\n"
        "  - canonical_name: new_game\n"
        "    execution: {type: press_key, key: f2}\n"
        "  - canonical_name: click_at\n"
        "    synonyms: ['kliknutí']\n"
        "    execution: {type: click_at}\n",
        encoding=encoding,
    )

    def locale_open(file, mode="r", **kwargs):
        kwargs.setdefault("encoding", "cp1250")
        return open(file, mode, **kwargs)

    monkeypatch.setattr(schema, "open", locale_open, raising=False)
    config = schema.load_action_config(path)
    assert config.canonical_names() == ["new_game", "click_at"]
    assert config.resolve_synonym("kliknutí") == "click_at"
