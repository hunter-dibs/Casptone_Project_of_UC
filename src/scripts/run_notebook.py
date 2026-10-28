#!/usr/bin/env python3
"""Execute this project's plain-Python notebook cells without Jupyter dependencies."""
from pathlib import Path
import contextlib
import io
import json
import os
import time

root=Path(__file__).resolve().parents[2]
path=root/'src/notebook/prepare_adni.ipynb'
nb=json.loads(path.read_text())
namespace={'__name__':'__main__'}
os.chdir(root)
count=0
for cell in nb['cells']:
    if cell['cell_type']!='code':continue
    count+=1
    stream=io.StringIO()
    started=time.monotonic()
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        exec(compile(''.join(cell['source']),f'{path.name}:cell{count}','exec'),namespace)
    cell['execution_count']=count
    cell['outputs']=[{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(keepends=True)}]
    print(f'Executed cell {count} ({time.monotonic()-started:.1f}s)')
out=path.with_name('prepare_adni.executed.ipynb')
out.write_text(json.dumps(nb,indent=1)+'\n')
print('Saved src/notebook/prepare_adni.executed.ipynb (local, git-ignored).')
