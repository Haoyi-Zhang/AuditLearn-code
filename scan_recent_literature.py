#!/usr/bin/env python3
from __future__ import annotations
import csv,re,time,urllib.parse,urllib.request,xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path(__file__).resolve().parent;PAPER=ROOT.parent/'paper';OUT=ROOT/'evidence'/'recent_literature_scan.csv'
QUERIES=[
 'all:"delayed feedback" AND (all:bandit OR all:"online learning")',
 'all:"reward dependent" AND all:delay AND all:bandit',
 'all:"outcome dependent" AND all:delay AND all:bandit',
 'all:"speculative decoding" AND (all:selection OR all:routing OR all:adaptive)',
 'all:"draft model" AND all:speculative AND (all:selection OR all:bandit)',
 'all:"triggered bandit" AND all:delay',
]
NS={'a':'http://www.w3.org/2005/Atom'}
def norm(s):return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
def bib_titles():
 text='\n'.join(p.read_text(errors='ignore') for p in PAPER.rglob('*.bib'))
 out=[]
 for m in re.finditer(r'(?ms)@\w+\s*\{\s*([^,]+),(.+?)(?=\n@|\Z)',text):
  tm=re.search(r'(?ms)\btitle\s*=\s*\{(.*?)\}\s*,',m.group(2))
  if tm:out.append((m.group(1).strip(),tm.group(1).replace('{','').replace('}','').replace('\n',' ')))
 return out
def fetch(q):
 params={'search_query':q,'start':0,'max_results':50,'sortBy':'submittedDate','sortOrder':'descending'}
 url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode(params)
 req=urllib.request.Request(url,headers={'User-Agent':'reviewer-literature-scan/1.0'})
 with urllib.request.urlopen(req,timeout=30) as r:data=r.read()
 root=ET.fromstring(data);rows=[]
 for e in root.findall('a:entry',NS):
  title=' '.join((e.findtext('a:title',default='',namespaces=NS)).split())
  summary=' '.join((e.findtext('a:summary',default='',namespaces=NS)).split())
  aid=e.findtext('a:id',default='',namespaces=NS);published=e.findtext('a:published',default='',namespaces=NS)
  authors='; '.join(x.findtext('a:name',default='',namespaces=NS) for x in e.findall('a:author',NS))
  rows.append({'query':q,'arxiv_id':aid.rsplit('/',1)[-1],'title':title,'authors':authors,'published':published,'url':aid,'summary':summary})
 return rows
def main():
 existing=bib_titles();rows=[]
 for i,q in enumerate(QUERIES):
  try:rows.extend(fetch(q))
  except Exception as e:rows.append({'query':q,'arxiv_id':'','title':'','authors':'','published':'','url':'','summary':'','error':type(e).__name__+': '+str(e)[:200]})
  time.sleep(3)
 uniq={}
 for r in rows:
  key=r.get('arxiv_id') or r.get('query')+'|ERROR';uniq.setdefault(key,r)
 rows=list(uniq.values())
 for r in rows:
  if not r.get('title'):r.update({'closest_bib_key':'','closest_bib_title':'','title_similarity':'','keyword_relevance':''});continue
  sims=[(SequenceMatcher(None,norm(r['title']),norm(t)).ratio(),k,t) for k,t in existing]
  sim,k,t=max(sims,default=(0,'',''))
  words=set(norm(r['title']+' '+r.get('summary','')).split())
  rel=len(words & {'delay','delayed','feedback','audit','bandit','speculative','decoding','draft','selection','routing','triggered','asynchronous'})
  r.update({'closest_bib_key':k,'closest_bib_title':t,'title_similarity':f'{sim:.3f}','keyword_relevance':rel})
 rows.sort(key=lambda r:(r.get('published',''),r.get('keyword_relevance',0)),reverse=True)
 OUT.parent.mkdir(exist_ok=True)
 fields=sorted({k for r in rows for k in r})
 with OUT.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(f'wrote {len(rows)} records to {OUT}')
if __name__=='__main__':main()
