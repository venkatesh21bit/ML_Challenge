import sys
import json

def convert(py_path, ipynb_path):
    with open(py_path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = text.split("# ════════════════════════════════════════════════════════════\n# CELL ")
    cells = []

    intro_text = parts[0].strip()
    if intro_text.startswith('"""') and intro_text.endswith('"""'):
        intro_text = intro_text[3:-3].strip()

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [intro_text + "\n"]
    })

    for p in parts[1:]:
        lines = p.split("\n")
        cell_body = "\n".join(lines[2:]).strip()
        if cell_body.startswith('"""') and cell_body.endswith('"""'):
            cell_body = cell_body[3:-3].strip()

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in cell_body.split("\n")]
        })

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Generated {ipynb_path} successfully!")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        convert(sys.argv[1], sys.argv[2])
    else:
        convert("notebooks/02_catboost_training.py", "notebooks/02_catboost_training.ipynb")
        convert("notebooks/02_hard_negative_mining_and_gpu_training.py", "notebooks/02_hard_negative_mining_and_gpu_training.ipynb")
        convert("notebooks/03_deberta_crossencoder.py", "notebooks/03_deberta_crossencoder.ipynb")
