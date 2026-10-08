from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parents[1];ROOT=P.parents[1];manifest=json.loads((P/'references/companion-sync.json').read_text());L=json.loads((P/'references/lessons.json').read_text())['lessons'];fail=[]
for row in manifest['chapters']:
 p=ROOT/row['notebook'];n=json.loads(p.read_text());meta=n['metadata']['companion']
 if meta['chapter']!=row['chapter'] or meta['source_sha256']!=row['source_sha256']:fail.append(row['chapter'])
 if not (ROOT/row['skill']).is_file():fail.append('skill '+str(row['chapter']))
 if sum(r['book_location']['chapter']==row['chapter'] for r in L)!=row['lessons']:fail.append('locations '+str(row['chapter']))
print(json.dumps({'status':'PASS' if not fail else 'FAIL','scope':'Public companion chapter bindings and conditional lessons, not full-book availability.','chapters':len(manifest['chapters']),'lessons':len(L),'failures':fail},indent=2));raise SystemExit(bool(fail))
