import json
from pathlib import Path


class MetadataLogger:
    def __init__(self, output_dir):
        self._output_dir = Path(output_dir)
        self._meta_dir = self._output_dir / "raw" / "metadata"
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._event_counter = 0

    def next_event_id(self):
        self._event_counter += 1
        return f"evt_{self._event_counter:06d}"

    def _append_jsonl(self, filename, data):
        path = self._meta_dir / filename
        with open(path, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_save_event(self, event):
        self._append_jsonl("save_events.jsonl", event)

    def log_track(self, track_data):
        self._append_jsonl("tracks.jsonl", track_data)

    def log_session(self, session_data):
        self._append_jsonl("sessions.jsonl", session_data)

    def ensure_classes_txt(self):
        classes_path = self._output_dir / "raw" / "classes.txt"
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        classes_path.write_text("vehicle\nplate\n")
