from pathlib import Path
import argparse,json
P=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--query',default='');a.add_argument('--lesson-id');a.add_argument('--workflow');a.add_argument('--modality');a.add_argument('--audience',choices=['beginner','competitor','practitioner'],default='competitor');a.add_argument('--grade');a.add_argument('--limit',type=int,default=5);args=a.parse_args();rows=json.loads((P/'references/lessons.json').read_text())['lessons']
rows=[r for r in rows if (not args.lesson_id or r['lesson_id']==args.lesson_id) and (not args.query or args.query.lower() in json.dumps(r).lower()) and (not args.workflow or args.workflow in r['dimensions']['workflow_stages']) and (not args.modality or args.modality in r['dimensions']['modalities']) and (not args.grade or r['scores'][args.audience]['grade']==args.grade)]
rows.sort(key=lambda r:(-r['scores'][args.audience]['score'],r['lesson_id']))
print(json.dumps({'matches':len(rows),'sort':'Existing editorial audience utility; not expected gain','lessons':rows[:max(0,args.limit)]},indent=2))
