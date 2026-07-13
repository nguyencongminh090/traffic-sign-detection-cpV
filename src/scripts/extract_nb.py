import json
import sys

def parse_notebook(filepath):
    print(f"\\n{'='*50}\\nNotebook: {filepath}\\n{'='*50}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            for i, cell in enumerate(nb.get('cells', [])):
                if cell['cell_type'] == 'markdown':
                    print(f"\\n--- Markdown Cell {i} ---")
                    print(''.join(cell.get('source', [])))
                elif cell['cell_type'] == 'code':
                    outputs = cell.get('outputs', [])
                    if outputs:
                        print(f"\\n--- Code Cell {i} Outputs ---")
                        for output in outputs:
                            if 'text' in output:
                                print(''.join(output['text']))
                            elif 'data' in output and 'text/plain' in output['data']:
                                print(''.join(output['data']['text/plain']))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        parse_notebook(arg)
