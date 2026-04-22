import csv
import os
from datetime import datetime, timezone

def log_feedback(
    question: str, answer: str, feedback: str, path: str = "feedback.csv"
) -> None:
    """Logs user feedback to a CSV file.

    Args:
        question (str): The original question.
        answer (str): The generated answer.
        feedback (str): 'thumbs_up' or 'thumbs_down'.
        path (str): Path to the feedback CSV file.
    """
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "feedback"])
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(), question, answer, feedback]
        )
