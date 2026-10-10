"""Reject book files from the public companion tree and Git history."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
protected=json.loads((ROOT/'audit/public-boundary.json').read_text())['excluded_book_sha256'];blocked=set(protected);fail=[];count=0
ignored={'.git','.venv','__pycache__','.pytest_cache','.ipynb_checkpoints'}
for p in ROOT.rglob('*'):
 if not p.is_file() or any(x in ignored for x in p.relative_to(ROOT).parts):continue
 count+=1;parts=p.relative_to(ROOT).parts
 if 'manuscript' in parts or 'book' in parts or p.name in ['book.md','book-sync.json','KaggleBook.md','latestbook.pdf'] or p.suffix.lower() in ['.epub','.pdf','.zip','.tar','.gz','.bundle']:
  fail.append(str(p.relative_to(ROOT))+': excluded book/archive path')
 if hashlib.sha256(p.read_bytes()).hexdigest() in blocked:fail.append(str(p.relative_to(ROOT))+': canonical book content')
if (ROOT/'.git').exists():
 # Every reachable historical blob is checked, not merely today's worktree.
 r=subprocess.run(['git','rev-list','--objects','--all'],cwd=ROOT,text=True,capture_output=True,check=True)
 for line in r.stdout.splitlines():
  oid,_,name=line.partition(' ')
  if not name:continue
  kind=subprocess.check_output(['git','cat-file','-t',oid],cwd=ROOT,text=True).strip()
  if kind!='blob':continue
  data=subprocess.check_output(['git','cat-file','blob',oid],cwd=ROOT)
  if hashlib.sha256(data).hexdigest() in blocked:fail.append('history '+name+': canonical book content')
  if '/references/book/' in name or name.endswith('/references/book.md') or name.startswith('manuscript/') or name.endswith(('.epub','.pdf','.bundle')):fail.append('history '+name+': excluded book path')
L=json.loads((ROOT/'skills/winning-kaggle-the-reproducible-way/references/lessons.json').read_text())['lessons']
if len(L)!=561 or len({r['lesson_id'] for r in L})!=561:fail.append('lesson coverage')
print(json.dumps({'status':'PASS' if not fail else 'FAIL','scope':'Book/archive paths, every protected canonical file hash (current and earlier editions) and reachable Git blob history. Companion examples and reviewed conditional lessons are intentionally included.','files':count,'lessons':len(L),'failures':fail},indent=2));raise SystemExit(bool(fail))
