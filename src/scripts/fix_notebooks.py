import json
import os

def process_notebook(file_path):
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        modified = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') in ['code', 'markdown']:
                new_source = []
                for line in cell.get('source', []):
                    original_line = line
                    
                    # Rule: replace /content/... with ./...
                    if '/content/' in line:
                        line = line.replace('/content/', './')
                        
                    # Rule: comment out colab specific imports/commands
                    if 'import google.colab' in line or 'from google.colab' in line:
                        line = '# ' + line
                    if 'drive.mount' in line:
                        line = '# ' + line
                        
                    if line != original_line:
                        modified = True
                    new_source.append(line)
                cell['source'] = new_source

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Saved changes to {file_path}")
        else:
            print(f"No changes needed for {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

process_notebook('data_overview.ipynb')
process_notebook('traffic_sign.ipynb')
