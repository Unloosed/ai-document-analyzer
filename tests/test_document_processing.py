import io
from unittest.mock import MagicMock, patch
from utils.document_processing import chunk_text, get_file_hash, read_txt, read_pdf, read_md

def test_chunk_text():
    text = "This is a test sentence that should be chunked into pieces."
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert all(len(c) <= 10 for c in chunks)

def test_chunk_text_empty():
    assert chunk_text("") == []

def test_read_txt():
    content = "Hello world"
    file = io.StringIO(content)
    assert read_txt(file) == content

def test_get_file_hash():
    content1 = b"test content"
    content2 = b"test content"
    content3 = b"different content"

    hash1 = get_file_hash(io.BytesIO(content1))
    hash2 = get_file_hash(io.BytesIO(content2))
    hash3 = get_file_hash(io.BytesIO(content3))

    assert hash1 == hash2
    assert hash1 != hash3

def test_read_txt_file_path(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("content")
    assert read_txt(str(p)) == "content"

def test_chunk_text_overlap_greater_than_size():
    text = "some text"
    chunks = chunk_text(text, chunk_size=5, overlap=10)
    assert len(chunks) > 0

def test_read_pdf():
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "page text"
    mock_reader.pages = [mock_page]
    with patch("utils.document_processing.PdfReader", return_value=mock_reader):
        assert read_pdf("dummy.pdf") == "page text"

def test_read_txt_bytes():
    content = b"Hello world"
    file = io.BytesIO(content)
    assert read_txt(file) == "Hello world"

def test_read_md():
    content = "# Title"
    file = io.StringIO(content)
    assert read_md(file) == content

def test_get_file_hash_path(tmp_path):
    p = tmp_path / "test.txt"
    p.write_bytes(b"content")
    assert get_file_hash(str(p)) == get_file_hash(io.BytesIO(b"content"))
