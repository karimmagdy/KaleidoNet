#!/usr/bin/env python3
"""Add from __future__ import annotations to all source files."""
import glob

files = sorted(set(
    glob.glob('kaleidonet/**/*.py', recursive=True) +
    ['kaleidonet/model.py']
))
files = [f for f in files if '__init__' not in f]

for fpath in files:
    with open(fpath, 'r') as f:
        content = f.read()
    if 'from __future__ import annotations' in content:
        print(f'  SKIP {fpath}')
        continue
    lines = content.split('\n')
    if lines[0].startswith('"""'):
        insert_idx = None
        for i in range(1, len(lines)):
            if '"""' in lines[i]:
                insert_idx = i + 1
                break
        if insert_idx is not None:
            lines.insert(insert_idx, '')
            lines.insert(insert_idx + 1, 'from __future__ import annotations')
        else:
            lines.insert(0, 'from __future__ import annotations')
            lines.insert(1, '')
    else:
        lines.insert(0, 'from __future__ import annotations')
        lines.insert(1, '')
    with open(fpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  DONE {fpath}')

print('All done!')
