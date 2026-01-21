import os
import time
import json
import shutil
from typing import Iterator, Optional, Dict, Any


class MockMailboxAdapter:
    """
    Polls a directory for .json email files and yields parsed email dictionaries.
    Copies the email JSON and referenced attachment files to a processed directory.
    """

    def __init__(
        self,
        inbox_dir: str,
        processed_dir: Optional[str] = None,
        poll_interval: float = 1.0,
    ):
        self.inbox_dir = inbox_dir
        self.processed_dir = processed_dir
        self.poll_interval = poll_interval

        os.makedirs(self.inbox_dir, exist_ok=True)
        if self.processed_dir:
            os.makedirs(self.processed_dir, exist_ok=True)

    def poll(self) -> Iterator[Dict[str, Any]]:
        """
        Continuously poll the inbox directory for .json email files.
        """
        while True:
            yield from self.fetch_once()
            time.sleep(self.poll_interval)

    def read_all_once(self) -> Iterator[Dict[str, Any]]:
        """
        Read and process all JSON emails in the inbox directory once.
        """
        for filename in sorted(os.listdir(self.inbox_dir)):
            if not filename.lower().endswith(".json"):
                continue

            json_path = os.path.join(self.inbox_dir, filename)

            try:
                email_data = self._load_json_email(json_path)
                yield email_data
                # self._mark_processed(json_path, email_data)
            except Exception as exc:
                print(f"Failed to process {filename}: {exc}")

    def fetch_once(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch all .json files currently in the inbox directory.
        """
        for filename in sorted(os.listdir(self.inbox_dir)):
            if not filename.lower().endswith(".json"):
                continue

            json_path = os.path.join(self.inbox_dir, filename)

            try:
                email_data = self._load_json_email(json_path)
                yield email_data
                # self._mark_processed(json_path, email_data)
            except Exception as exc:
                print(f"Failed to process {filename}: {exc}")

    def _load_json_email(self, path: str) -> Dict[str, Any]:
        """
        Load a JSON email file.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _mark_processed(self, json_path: str, email_data: Dict[str, Any]) -> None:
        """
        Copy JSON email and all attachment files to processed directory,
        then remove originals from inbox.
        """
        if not self.processed_dir:
            os.remove(json_path)
            return

        # Copy JSON email
        json_dest = os.path.join(self.processed_dir, os.path.basename(json_path))
        shutil.copy2(json_path, json_dest)

        # Copy attachment files
        for attachment in email_data.get("attachments", []):
            file_name = attachment.get("file_name")
            if not file_name:
                continue

            src = os.path.join(self.inbox_dir, file_name)
            dst = os.path.join(self.processed_dir, file_name)

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: attachment not found: {file_name}")

        # Remove original JSON and attachments from inbox
        os.remove(json_path)
        for attachment in email_data.get("attachments", []):
            file_name = attachment.get("file_name")
            if not file_name:
                continue
            src = os.path.join(self.inbox_dir, file_name)
            if os.path.exists(src):
                os.remove(src)


if __name__ == "__main__":
    adapter = MockMailboxAdapter(
        inbox_dir="./inbox",
        processed_dir="./processed",
        poll_interval=2.0,
    )

    for i, email in enumerate(adapter.read_all_once()):
        print("Received email subject:", email.get("subject"))
    print(f'Processed {i + 1} emails.')