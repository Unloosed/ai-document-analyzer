import os
from unittest.mock import MagicMock, patch
import pytest
from store import add_document, search_docs, get_stats
from rag import answer_question
from embeddings import OpenAIEmbeddingFunction
from feedback import log_feedback

@pytest.fixture
def mock_chroma():
    with patch("store.collection") as mock_col:
        yield mock_col

@pytest.fixture
def mock_openai_chat():
    with patch("rag.chat_client") as mock_client:
        yield mock_client

@pytest.fixture
def mock_openai_embeddings():
    with patch("embeddings.embedding_client") as mock_client:
        yield mock_client

@pytest.fixture
def mock_manifest():
    with patch("store._load_manifest") as mock_load, \
         patch("store._save_manifest") as mock_save:
        mock_load.return_value = {"hashes": {}, "total_chunks": 0}
        yield mock_load, mock_save

def test_add_document_duplicate(mock_manifest, mock_chroma):
    mock_load, mock_save = mock_manifest
    mock_load.return_value = {"hashes": {"hash123": "test.txt"}, "total_chunks": 1}

    result = add_document("test.txt", "hash123", ["chunk1"])
    assert result is False
    mock_chroma.add.assert_not_called()

def test_add_document_new(mock_manifest, mock_chroma):
    mock_load, mock_save = mock_manifest

    result = add_document("test.txt", "hash123", ["chunk1"])
    assert result is True
    mock_chroma.add.assert_called_once()
    mock_save.assert_called_once()

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

def test_answer_question(mock_openai_chat, mock_chroma):
    # Mock store.search_docs called via rag.answer_question
    with patch("rag.search_docs") as mock_search:
        mock_search.return_value = [("doc1", {"source": "src1", "chunk_index": 0}, 0.1)]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is the answer [1]"
        mock_openai_chat.chat.completions.create.return_value = mock_response

        answer, retrieved = answer_question("What is it?")
        assert "This is the answer [1]" in answer
        assert len(retrieved) == 1

def test_answer_question_strict_no_results(mock_chroma):
    with patch("rag.search_docs") as mock_search:
        mock_search.return_value = []
        answer, retrieved = answer_question("What is it?", strict_mode=True)
        assert "couldn't find enough relevant information" in answer
        assert len(retrieved) == 0

def test_answer_question_confidence_filter(mock_openai_chat, mock_chroma):
    with patch("rag.search_docs") as mock_search:
        mock_search.return_value = [
            ("doc1", {"source": "s1", "chunk_index": 0}, 0.1),
            ("doc2", {"source": "s2", "chunk_index": 0}, 0.9)
        ]
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Ans"
        mock_openai_chat.chat.completions.create.return_value = mock_response

        # Distance 0.9 should be filtered out if min_confidence is 0.5 (threshold 1-0.5 = 0.5)
        _, retrieved = answer_question("query", min_confidence=0.5)
        assert len(retrieved) == 1
        assert retrieved[0][0] == "doc1"

def test_log_feedback(tmp_path):
    feedback_file = tmp_path / "feedback.csv"
    log_feedback("Q", "A", "thumbs_up", path=str(feedback_file))
    assert feedback_file.exists()
    content = feedback_file.read_text()
    assert "timestamp,question,answer,feedback" in content
    assert "Q,A,thumbs_up" in content

def test_get_stats(mock_manifest):
    mock_load, _ = mock_manifest
    mock_load.return_value = {
        "hashes": {"h1": "f1", "h2": "f2"},
        "total_chunks": 10
    }
    stats = get_stats()
    assert stats["total_chunks"] == 10
    assert stats["unique_documents"] == 2

def test_openai_embedding_function(mock_openai_embeddings):
    mock_openai_embeddings.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2])]
    eb = OpenAIEmbeddingFunction()
    res = eb(["test"])
    assert len(res) == 1
    assert list(res[0]) == [0.1, 0.2]
