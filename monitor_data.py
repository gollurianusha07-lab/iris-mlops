
import time
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "iris.csv"
TRAIN_SCRIPT = BASE_DIR / "train_pipeline.py"


class DatasetHandler(FileSystemEventHandler):

    def __init__(self):
        self.last_run = 0

    def retrain(self):
        # Avoid retraining repeatedly for rapid file events.
        if time.time() - self.last_run < 3:
            return

        self.last_run = time.time()
        print("\nDataset changed! Starting retraining...")

        try:
            result = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT)],
                cwd=str(BASE_DIR),
                check=False,
            )

            if result.returncode == 0:
                print("Retraining completed successfully!")
            else:
                print("Retraining failed. Check the error above.")

        except Exception as error:
            print("Error during retraining:", error)

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).resolve() == DATA_FILE.resolve():
            self.retrain()

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).resolve() == DATA_FILE.resolve():
            self.retrain()

    def on_moved(self, event):
        if not event.is_directory and Path(event.dest_path).resolve() == DATA_FILE.resolve():
            self.retrain()


if __name__ == "__main__":
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    observer = Observer()
    observer.schedule(DatasetHandler(), str(BASE_DIR), recursive=False)
    observer.start()

    print("Monitoring iris.csv for changes...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
