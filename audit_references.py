#!/usr/bin/env python3
from __future__ import annotations
import csv, json, re, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent/'paper'
OUT=ROOT/'evidence'/'reference_network_audit.csv'

def norm(s:str)->str:
    s=re.sub(r'\\[A-Za-z]+\s*\{([^{}]*)\}',r'\1',s)
    s=s.replace('{','').replace('}','')
    return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()

def entries():
    text='\n'.join(p.read_text(errors='ignore') for p in PAPER.rglob('*.bib'))
    starts=list(re.finditer(r'@(?P<kind>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,',text))
    for i,m in enumerate(starts):
        block=text[m.start(): starts[i+1].start() if i+1<len(starts) else len(text)]
        fields={k.lower():v.strip().strip('{}"') for k,v in re.findall(r'(?ms)^\s*(\w+)\s*=\s*[\{"](.+?)[\}"]\s*,?\s*$',block)}
        # fallback non-greedy field parser for one-line values
        for fm in re.finditer(r'(?ms)\b(doi|url|title|year|author|journal|booktitle)\s*=\s*\{(.*?)\}\s*,?',block):
            fields.setdefault(fm.group(1).lower(),fm.group(2).replace('\n',' ').strip())
        yield m.group('key'),fields

def fetch_crossref(doi:str):
    url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
    req=urllib.request.Request(url,headers={'User-Agent':'research-artifact-reference-audit/1.0 (mailto:invalid@example.invalid)'})
    with urllib.request.urlopen(req,timeout=15) as r:
        data=json.load(r)['message']
    title=' '.join(data.get('title') or [])
    years=[]
    for f in ('published-print','published-online','issued'):
        try: years.append(str(data[f]['date-parts'][0][0]))
        except Exception: pass
    return {'resolved_url':data.get('URL',''),'remote_title':title,'remote_year':'/'.join(dict.fromkeys(years)),'http_status':200,'source':'Crossref'}

def fetch_url(url:str):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 research-artifact-audit'})
    with urllib.request.urlopen(req,timeout=15) as r:
        final=r.geturl(); status=getattr(r,'status',200)
        ctype=r.headers.get('content-type','')
        chunk=r.read(120000) if 'text' in ctype or 'html' in ctype else b''
    title=''
    if chunk:
        m=re.search(br'(?is)<title[^>]*>(.*?)</title>',chunk)
        if m: title=re.sub(r'\s+',' ',m.group(1).decode('utf-8','ignore')).strip()
    return {'resolved_url':final,'remote_title':title,'remote_year':'','http_status':status,'source':'URL'}

def one(item):
    key,f=item
    doi=f.get('doi','').strip()
    url=f.get('url','').strip()
    base={'key':key,'bib_title':f.get('title',''),'bib_year':f.get('year',''),'doi':doi,'url':url,
          'source':'','http_status':'','resolved_url':'','remote_title':'','remote_year':'','title_similarity':'','year_consistent':'','error':''}
    try:
        got=fetch_crossref(doi) if doi else fetch_url(url) if url else None
        if got:
            base.update(got)
            if got['remote_title']:
                base['title_similarity']=f"{SequenceMatcher(None,norm(base['bib_title']),norm(got['remote_title'])).ratio():.3f}"
            if base['bib_year'] and got['remote_year']:
                base['year_consistent']=str(base['bib_year'] in got['remote_year'])
        else:
            base['error']='no DOI or URL in BibTeX'
    except Exception as e:
        base['error']=type(e).__name__+': '+str(e)[:240]
    return base

def main():
    items=list(entries())
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(one,x):x[0] for x in items}
        for fut in as_completed(futs): rows.append(fut.result())
    rows.sort(key=lambda r:r['key'].lower())
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ['key']
    with OUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={
        'entries':len(rows),'resolved':sum(bool(r['http_status']) for r in rows),
        'errors':sum(bool(r['error']) for r in rows),
        'low_title_similarity':sum(bool(r['title_similarity']) and float(r['title_similarity'])<0.55 for r in rows),
        'year_mismatches':sum(r['year_consistent']=='False' for r in rows),
    }
    (OUT.parent/'reference_network_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
