import os
from unittest.mock import MagicMock, patch
import pytest
from rag import add_document, search_docs, answer_question, get_stats

@pytest.fixture
def mock_chroma():
    with patch("rag.collection") as mock_col:
        yield mock_col

@pytest.fixture
def mock_openai():
    with patch("rag.client") as mock_client:
        yield mock_client

def test_add_document_duplicate(mock_chroma):
    mock_chroma.get.return_value = {"ids": ["123"]}
    result = add_document("test.txt", "hash123", ["chunk1"])
    assert result is False
    mock_chroma.add.assert_not_called()

def test_add_document_new(mock_chroma):
    mock_chroma.get.return_value = {"ids": []}
    result = add_document("test.txt", "hash123", ["chunk1"])
    assert result is True
    mock_chroma.add.assert_called_once()

def test_search_docs(mock_chroma):
    mock_chroma.query.return_value = {
        "documents": [["doc1"]],
        "metadatas": [[{"source": "src1"}]],
        "distances": [[0.1]]
    }
    results = search_docs("query")
    assert len(results) == 1
    assert results[0][0] == "doc1"
    assert results[0][1]["source"] == "src1"

def test_answer_question(mock_openai, mock_chroma):
    mock_chroma.query.return_value = {
        "documents": [["doc1"]],
        "metadatas": [[{"source": "src1", "chunk_index": 0}]],
        "distances": [[0.1]]
    }

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is the answer"
    mock_openai.chat.completions.create.return_value = mock_response

    answer, retrieved = answer_question("What is it?")
    assert answer == "This is the answer"
    assert len(retrieved) == 1

def test_answer_question_strict_no_results(mock_openai, mock_chroma):
    mock_chroma.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }
    answer, retrieved = answer_question("What is it?", strict_mode=True)
    assert "couldn't find enough relevant information" in answer
    assert len(retrieved) == 0

def test_log_feedback(tmp_path):
    feedback_file = tmp_path / "feedback.csv"
    from rag import log_feedback
    log_feedback("Q", "A", "thumbs_up", path=str(feedback_file))
    assert feedback_file.exists()
    content = feedback_file.read_text()
    assert "timestamp,question,answer,feedback" in content
    assert "Q,A,thumbs_up" in content

def test_get_stats(mock_chroma):
    mock_chroma.count.return_value = 10
    mock_chroma.get.return_value = {"metadatas": [{"file_hash": "h1"}, {"file_hash": "h1"}, {"file_hash": "h2"}]}
    stats = get_stats()
    assert stats["total_chunks"] == 10
    assert stats["unique_documents"] == 2
