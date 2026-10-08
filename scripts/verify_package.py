"""Check reader links, skill resources and execute every shipped notebook."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

import nbformat
import yaml
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
        self.ids=set()
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        for field in ['href','src']:
            if attrs.get(field):
                self.links.append(attrs[field])


def execute(path):
    notebook=nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=90,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
    nbformat.write(notebook,path)
    errors=[out for cell in notebook.cells if cell.cell_type=='code' for out in cell.get('outputs',[]) if out.output_type=='error']
    if errors:
        raise AssertionError(f'{path.name}: notebook errors')
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'code_cells':sum(cell.cell_type=='code' for cell in notebook.cells),'execution':'nbclient fresh Jupyter kernel','errors':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-execution',action='store_true')
    args=parser.parse_args()
    chapter_map=json.loads((ROOT/'chapter-map.json').read_text())
    if len(chapter_map)!=65:
        raise AssertionError('expected 65 chapter bindings')
    failures,external=[],set()
    pages=list((ROOT/'site').rglob('*.html'))
    for page in pages:
        text=page.read_text()
        parsed=Links();parsed.feed(text)
        if '\u2014' in text:
            failures.append(f'{page.name}: em dash')
        for link in parsed.links:
            url=urlsplit(link)
            if url.scheme or url.netloc:
                external.add(link);continue
            target=(page.parent/unquote(url.path)).resolve() if url.path else page
            if target.is_dir():
                target=target/'index.html'
            if not target.exists():
                failures.append(f'{page.relative_to(ROOT)}: missing {link}')
            elif url.fragment and target.suffix=='.html':
                target_links=Links();target_links.feed(target.read_text())
                if unquote(url.fragment) not in target_links.ids:
                    failures.append(f'{page.name}: missing fragment {link}')
            if url.path.endswith(('.md','.ipynb')):
                failures.append(f'{page.name}: raw asset served on site')
    skills=list((ROOT/'skills').glob('*/SKILL.md'))
    if len(skills)!=66:
        failures.append(f'expected 66 skills, found {len(skills)}')
    for path in skills:
        text=path.read_text()
        metadata=yaml.safe_load(text.split('---',2)[1])
        if not isinstance(metadata,dict) or not re.fullmatch(r'[a-z0-9-]{1,64}',metadata.get('name','')) or not isinstance(metadata.get('description'),str) or not metadata['description'].strip():
            failures.append(f'{path}: skill metadata invalid')
        for _,link in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',text):
            if not urlsplit(link).scheme and not (path.parent/link).resolve().exists():
                failures.append(f'{path}: missing resource {link}')
    if failures:
        raise AssertionError('\n'.join(failures))
    notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
    if len(notebooks)!=65:
        raise AssertionError(f'expected 65 notebooks, found {len(notebooks)}')
    previous_path=ROOT/'audit/package-verification.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
    receipt={'chapters':65,'reader_pages':len(pages),'skills':len(skills),'local_link_failures':[],
             'external_links':sorted(external),'notebook_execution':'NOT_RUN' if args.skip_execution else 'RUNNING'}
    if args.skip_execution and previous.get('notebook_execution')=='PASS' and all((ROOT/item['path']).exists() and hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in previous.get('notebooks',[])) and len(previous.get('notebooks',[]))==65:
        receipt['notebook_execution']='PASS'
        receipt['notebooks']=previous['notebooks']
    (ROOT/'audit/package-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not args.skip_execution:
        # Temporary project-contained kernelspec selects this exact interpreter.
        # No global Jupyter installation or user registry mutation is needed.
        with tempfile.TemporaryDirectory(prefix='kaggle-notebooks-') as directory:
            spec=Path(directory)/'kernels/python3'
            spec.mkdir(parents=True)
            (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                 'display_name':'Python 3','language':'python'}))
            os.environ['JUPYTER_PATH']=directory
            os.environ['JUPYTER_RUNTIME_DIR']=str(Path(directory)/'runtime')
            os.environ['IPYTHONDIR']=str(Path(directory)/'ipython')
            os.environ['MPLCONFIGDIR']=str(Path(directory)/'matplotlib')
            os.environ['OPENBLAS_NUM_THREADS']='1'
            with ThreadPoolExecutor(max_workers=3) as pool:
                notebook_receipts=list(pool.map(execute,notebooks))
        receipt['notebook_execution']='PASS'
        receipt['notebooks']=notebook_receipts
        (ROOT/'audit/package-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['external_links','notebooks']},indent=2))


if __name__=='__main__':
    main()
