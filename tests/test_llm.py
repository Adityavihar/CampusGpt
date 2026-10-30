import pytest
from unittest.mock import MagicMock, patch
from services.llm import generate_answer


def test_generate_answer_empty_question():
    res = generate_answer("", "Some context")
    assert "Please ask a question" in res


def test_generate_answer_empty_context():
    res = generate_answer("What is the fee?", "")
    assert "I couldn't find this information" in res


@patch("services.llm._get_client")
def test_generate_answer_mocked(mock_get_client):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "VIT-AP offers B.Tech in CSE [VITAP_Programs.pdf, Page 1]."
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    mock_get_client.return_value = mock_client

    context = "UNIVERSITY DOCUMENTS:\nSource: VITAP_Programs.pdf\nPage: 1\nContent: B.Tech in CSE"
    ans = generate_answer("What B.Tech programs?", context)
    assert "B.Tech in CSE" in ans
