#!/usr/bin/env python3
"""Render executed chapter notebooks as a portable website workbook.

Does not execute notebooks, certify accuracy, or upload anything.
"""
from pathlib import Path
import argparse,base64,hashlib,html,json,re
from urllib.parse import urlsplit

def text(value):return ''.join(value) if isinstance(value,list) else str(value)
def escape(value):return html.escape(text(value),quote=True)
def markdown(value):
 try:
  from markdown_it import MarkdownIt
  return MarkdownIt('commonmark',{'html':False}).render(text(value))
 except ImportError:
  import mistune
  return mistune.create_markdown(escape=True)(text(value))
def flatten(value,prefix=''):
 if isinstance(value,dict):return [(k,v) for key,item in value.items() for k,v in flatten(item,f'{prefix} / {key}' if prefix else key)]
 if isinstance(value,list):return [(k,v) for i,item in enumerate(value) for k,v in flatten(item,f'{prefix} / {i+1}')]
 return [(prefix.replace('_',' '),value)]
def output_html(output):
 if output['output_type']=='error':raise ValueError('Notebook contains an error output')
 data=output.get('data',{})
 if 'image/png' in data:
  image=text(data['image/png']);base64.b64decode(image,validate=True)
  return f'<img alt="Saved figure from this chapter calculation" src="data:image/png;base64,{image}">'
 value=data.get('application/json',text(output.get('text',data.get('text/plain',''))))
 try:value=json.loads(value) if isinstance(value,str) else value
 except ValueError:return '<pre>'+escape(value)+'</pre>'
 if isinstance(value,(dict,list)):
  return '<div class="table-scroll"><table><thead><tr><th scope="col">Quantity or decision</th><th scope="col">Result</th></tr></thead><tbody>'+''.join('<tr><td>'+escape(k)+'</td><td>'+escape(v)+'</td></tr>' for k,v in flatten(value))+'</tbody></table></div>'
 return '<pre>'+escape(value)+'</pre>'
def validate(notebook):
 for cell in notebook['cells']:
  if cell['cell_type']=='code' and text(cell['source']).strip():
   if cell.get('execution_count') is None:raise ValueError('Notebook has an unexecuted code cell')
   if any(out['output_type']=='error' for out in cell.get('outputs',[])):raise ValueError('Notebook contains an error output')
def build(args):
 files=sorted(args.notebooks.glob('*.ipynb'))
 if not files:raise ValueError('No chapter notebooks found')
 rows=[];seen=set()
 for p in files:
  n=json.loads(p.read_text());validate(n);meta=n.get('metadata',{}).get('companion',{});chapter=meta.get('chapter',p.stem)
  if chapter in seen:raise ValueError('Duplicate chapter identity')
  seen.add(chapter);rows.append((p,n,chapter,meta))
 mapping=json.loads(args.chapter_map.read_text()) if args.chapter_map else []
 bindings={r['number']:r for r in mapping}
 if mapping and set(bindings)!=seen:raise ValueError('Notebook/chapter mapping coverage differs')
 lessons=json.loads(args.lessons.read_text())['lessons'] if args.lessons else []
 if lessons and (len({r['lesson_id'] for r in lessons})!=len(lessons) or any(r['book_location']['chapter'] not in seen for r in lessons)):raise ValueError('Lesson locations incomplete or duplicated')
 for p,n,chapter,meta in rows:
  if chapter in bindings and meta.get('source_sha256')!=bindings[chapter]['source_sha256']:raise ValueError(f'{p.name}: stale chapter binding')
 args.output.mkdir(parents=True,exist_ok=True);receipts=[];cards=[]
 def page(title,body,notebook=None):
  links='<a href="../index.html">All chapters</a><a href="index.html">Workbook</a>'
  if notebook:links+=f'<a href="{escape(args.repo_url)}/blob/main/notebooks/{escape(notebook)}">Notebook on GitHub</a>'
  return '<!doctype html><html lang="en" class="cmp-page cmp-home"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title><link rel="stylesheet" href="'+escape(args.stylesheet)+'"><style>main{max-width:1000px;margin:auto;padding:32px 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;height:auto}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px;overflow-wrap:anywhere}details{margin:20px 0;padding:12px;border:1px solid #ddd}textarea{display:block;width:100%;min-height:100px;box-sizing:border-box;font:inherit}label{display:block;margin:20px 0}a,summary,button{cursor:pointer}button{font:inherit;padding:8px}section{margin:28px 0}</style></head><body><header class="cmp-header"><div class="cmp-topbar"><nav class="cmp-nav">'+links+'</nav></div></header><main class="cmp-wrap">'+body+'</main><footer class="cmp-footer"><div class="cmp-footer-inner"><p>'+escape(args.title)+'</p><a href="'+escape(args.book_url)+'">About the book</a> · <a href="'+escape(args.repo_url)+'">GitHub</a></div></footer></body></html>'
 for p,n,chapter,meta in rows:
  binding=bindings.get(chapter,{})
  if binding and meta.get('source_sha256')!=binding['source_sha256']:raise ValueError(f'{p.name}: stale chapter binding')
  title=binding.get('title',p.stem);question=binding.get('question','Predict the outcome of the worked calculation, then compare it with its saved result.')
  body='<h1>Chapter '+escape(chapter)+': '+escape(title)+'</h1><section><h2>Try the decision first</h2><p>'+escape(question)+'</p><p>Write your expectation before opening the worked results. Use the chapter activity to compare supported inputs. To change the calculation itself, run the notebook on GitHub.</p>'
  fields=['Prediction and reason','Assessment population and eligible information','Comparison and evidence required','Keep, reject or defer, with limits']
  for i,label in enumerate(fields):body+=f'<label>{escape(label)}<textarea data-answer="{i}"></textarea></label>'
  body+='<button id="export" type="button">Download my answers</button><p>Your answers stay on this page until you download them. They are not sent to a server.</p></section><p><a href="../'+escape(p.stem)+'/reader.html">Open the chapter activity</a></p><details><summary>Worked notebook and saved results</summary>'
  for cell in n['cells']:
   if cell['cell_type']=='markdown':body+=markdown(cell['source'])
   elif cell['cell_type']=='code':
    body+='<details><summary>Calculation code</summary><pre><code>'+escape(cell['source'])+'</code></pre></details>'
    body+=''.join(output_html(out) for out in cell.get('outputs',[]))
  body+='</details>'
  matched=[r for r in lessons if r['book_location']['chapter']==chapter]
  if matched:
   body+='<section><h2>Apply a lesson from this chapter</h2><p>Select a lesson whose conditions match your project. State how you would test it and what would make you reject it.</p>'
   for r in matched:
    body+='<details><summary>'+escape(r['title'])+'</summary>'
    for label,key in [('Conditions','trigger_conditions'),('Action','action_steps'),('Check','verification_steps'),('Limits','limitations')]:body+='<h3>'+label+'</h3><ul>'+''.join('<li>'+escape(item)+'</li>' for item in r[key])+'</ul>'
    body+='<p><a href="../lessons.html">Find source links and value grades in the lesson atlas</a></p></details>'
   body+='</section>'
  body+='<script>document.getElementById("export").addEventListener("click",()=>{const answers=[...document.querySelectorAll("textarea[data-answer]")].map(x=>({prompt:x.parentElement.firstChild.textContent.trim(),answer:x.value}));const blob=new Blob([JSON.stringify({page:location.pathname,answers},null,2)],{type:"application/json"});const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download="workbook-answers.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});</script>'
  name=p.stem+'.html';(args.output/name).write_text(page(title,body,p.name));cards.append('<li><a href="'+escape(name)+'">Chapter '+escape(chapter)+': '+escape(title)+'</a> · '+str(len(matched))+' lessons</li>');receipts.append({'chapter':chapter,'notebook':p.name,'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':meta.get('source_sha256'),'page':name,'lessons':len(matched),'code_cells':sum(c['cell_type']=='code' for c in n['cells'])})
 intro='<h1>'+escape(args.title)+': workbook</h1><p>Predict a result, inspect the worked notebook, then apply a source-backed lesson to your own modeling decision.</p><p>'+str(len(rows))+' chapter workbooks · '+str(len(lessons))+' conditional lessons</p><ol>'+''.join(cards)+'</ol>'
 (args.output/'index.html').write_text(page(args.title+' workbook',intro));receipt={'scope':'Executed-notebook presentation and source binding; not an independent accuracy audit or publication receipt.','chapters':receipts,'lessons':len(lessons)};(args.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--notebooks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--title',required=True);p.add_argument('--repo-url',required=True);p.add_argument('--book-url',required=True);p.add_argument('--stylesheet',default='../companion.css');p.add_argument('--chapter-map',type=Path);p.add_argument('--lessons',type=Path);a=p.parse_args()
 for url in [a.repo_url,a.book_url]:
  if urlsplit(url).scheme!='https':p.error('Public links must use https')
 r=build(a);print(f"Rendered {len(r['chapters'])} chapter workbooks and {r['lessons']} lessons")
if __name__=='__main__':main()
