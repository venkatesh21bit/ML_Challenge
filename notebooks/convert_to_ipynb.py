import json
import re
import os

def py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    # Split by cell header
    blocks = re.split(r'# ═+\s*# CELL \d+[^\n]*\s*# ═+', content)
    
    # blocks[0] is header/docstring
    if blocks[0].strip():
        doc = blocks[0].strip().strip('"""').strip("'''").strip()
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': [line + '\n' for line in doc.split('\n')]
        })
    
    headers = re.findall(r'# CELL \d+[^\n]*', content)
    for i, header in enumerate(headers):
        if i + 1 < len(blocks):
            cell_code = blocks[i+1].strip()
            if cell_code.startswith('"""') and cell_code.endswith('"""'):
                cell_code = cell_code[3:-3].strip()
            elif cell_code.startswith("'''") and cell_code.endswith("'''"):
                cell_code = cell_code[3:-3].strip()
            
            cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [line + '\n' for line in cell_code.split('\n')]
            })

    nb = {
        'cells': cells,
        'metadata': {
            'language_info': {'name': 'python'}
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {ipynb_path}")

if __name__ == '__main__':
    py_to_ipynb('notebooks/01_candidate_generation_and_benchmark.py', 'notebooks/01_candidate_generation_and_benchmark.ipynb')
    py_to_ipynb('notebooks/01_blocking_benchmark.py', 'notebooks/01_blocking_benchmark.ipynb')
    py_to_ipynb('notebooks/02_catboost_training.py', 'notebooks/02_catboost_training.ipynb')
    py_to_ipynb('notebooks/03_deberta_crossencoder.py', 'notebooks/03_deberta_crossencoder.ipynb')
    py_to_ipynb('notebooks/04_ensemble_final.py', 'notebooks/04_ensemble_final.ipynb')
