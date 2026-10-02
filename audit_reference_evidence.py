#!/usr/bin/env python3
from __future__ import annotations
import csv,json,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PAPER=ROOT.parent/'paper'
def parse_bib():
 text='\n'.join(p.read_text(errors='ignore') for p in PAPER.rglob('*.bib'))
 starts=list(re.finditer(r'@(?P<kind>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,',text));out={}
 for i,m in enumerate(starts):
  block=text[m.start():starts[i+1].start() if i+1<len(starts) else len(text)]
  fields={}
  for f in ('title','author','year','doi','url','journal','booktitle','publisher'):
   mm=re.search(r'(?ms)\b'+f+r'\s*=\s*[\{"](.*?)[\}"]\s*,',block)
   if mm:fields[f]=re.sub(r'\s+',' ',mm.group(1)).strip()
  out[m.group('key')]=fields
 return out
def find_csv(name):
 hits=list(ROOT.rglob(name));return hits[0] if hits else None
def rows(p):
 if not p:return []
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 bib=parse_bib();errors=[];warnings=[]
 for k,f in bib.items():
  for req in ('title','author','year'):
   if not f.get(req):errors.append(f'{k}: missing {req}')
 dois=[re.sub(r'^https?://(?:dx\.)?doi\.org/','',f.get('doi','').lower()).rstrip('.') for f in bib.values() if f.get('doi')]
 dupdoi=[x for x,n in Counter(dois).items() if n>1]
 if dupdoi:errors.append('duplicate DOI values: '+', '.join(dupdoi))
 auditp=find_csv('reference_audit.csv');audit=rows(auditp)
 # Find likely key column.
 audit_keys=set()
 if audit:
  cols=audit[0].keys();kc=next((c for c in cols if c.lower() in ('key','bibkey','citation_key','bibtex_key')),None)
  if kc:audit_keys={r[kc].strip() for r in audit if r.get(kc)}
  else:warnings.append('reference_audit.csv has no recognized key column')
 else:errors.append('reference_audit.csv missing or empty')
 if audit_keys:
  miss=set(bib)-audit_keys;extra=audit_keys-set(bib)
  if miss:errors.append(f'{len(miss)} bibliography keys missing from reference audit: '+', '.join(sorted(miss)[:20]))
  if extra:warnings.append(f'{len(extra)} audit keys are not retained bibliography entries')
 netp=find_csv('reference_network_audit.csv');net=rows(netp);netkeys={r.get('key','').strip() for r in net}
 if set(bib)-netkeys:warnings.append(f'{len(set(bib)-netkeys)} bibliography entries lack a network-audit row')
 resolved=sum(bool(r.get('http_status')) for r in net);low=[]
 for r in net:
  try:
   if r.get('title_similarity') and float(r['title_similarity'])<0.45:low.append(r.get('key'))
  except:pass
 if low:warnings.append(f'{len(low)} network title similarities below 0.45 require manual inspection: '+', '.join(low[:20]))
 passages=rows(find_csv('source_passages.csv'))
 report={'status':'PASS' if not errors else 'FAIL','bib_entries':len(bib),'reference_audit_rows':len(audit),'network_audit_rows':len(net),'network_resolved':resolved,'passage_rows':len(passages),'errors':errors,'warnings':warnings}
 (ROOT/'evidence'/'reference_evidence_gate.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));return 1 if errors else 0
if __name__=='__main__':raise SystemExit(main())
