import os

from agentops_loop.settings import AgentSettings


def test_llm_credentials_from_dotenv_are_exposed_to_sdk(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_API_VERSION", raising=False)
    (tmp_path / ".env").write_text(
        "AZURE_OPENAI_API_KEY=dotenv-key\n"
        "AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/\n"
        "OPENAI_API_VERSION=2025-04-01-preview\n",
        encoding="utf-8",
    )

    AgentSettings()

    assert os.environ["AZURE_OPENAI_API_KEY"] == "dotenv-key"
    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://example.openai.azure.com/"
    assert os.environ["OPENAI_API_VERSION"] == "2025-04-01-preview"


def test_shell_credentials_take_precedence_over_dotenv(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "shell-key")
    (tmp_path / ".env").write_text("AZURE_OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")

    AgentSettings()

    assert os.environ["AZURE_OPENAI_API_KEY"] == "shell-key"