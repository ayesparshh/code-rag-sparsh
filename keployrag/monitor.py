import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from keployrag.index import add_to_index, save_index, delete_entities
from keployrag.embeddings import generate_embeddings
from keployrag.config import WATCHED_DIR, IGNORE_PATHS

def should_ignore_path(path):
    """Check if the given path should be ignored based on the IGNORE_PATHS list."""
    normalized_path = os.path.normpath(path)
    for ignore_path in IGNORE_PATHS:
        normalized_ignore = os.path.normpath(ignore_path)
        if normalized_ignore in normalized_path:
            return True
        if "__pycache__" in normalized_path:
            return True
    return False

class CodeChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory or should_ignore_path(event.src_path):
            return

        if event.src_path.endswith(".py"):
            print(f"Detected change in file: {event.src_path}")
            try:
                relative_path = os.path.relpath(event.src_path, WATCHED_DIR)
                delete_expr = f'filepath == "{relative_path}"'
                delete_entities(filter_expr=delete_expr)
                print(f"Deleted old entries for file: {event.src_path}")

                with open(event.src_path, 'r', encoding='utf-8') as f:
                    full_content = f.read()
                embeddings = generate_embeddings(full_content)
                if embeddings is not None and len(embeddings) > 0:
                    filename = os.path.basename(event.src_path)
                    add_to_index(embeddings, full_content, filename, event.src_path)
                    save_index()
                    print(f"Updated index with new content for file: {event.src_path}")
            except Exception as e:
                print(f"Error updating index for {event.src_path}: {str(e)}")

def start_monitoring():
    event_handler = CodeChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCHED_DIR, recursive=True)
    observer.start()
    print(f"Started monitoring {WATCHED_DIR}... now run the streamlit server")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
