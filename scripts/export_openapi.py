"""Export the exact backend contract without loading models or Gradio."""
import json
import os
from pathlib import Path

os.environ["CHURN_ENABLE_UI"] = "0"
from app.main import app


def main():
    destination = Path(__file__).resolve().parents[1] / "docs/openapi.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
    print(f"Exported {destination}")


if __name__ == "__main__":
    main()
