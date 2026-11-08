"""Verify release shards and restore measured data without overwriting files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['NR-I','NR-II','FR','all'],default='all')
    p.add_argument('--raw',action='store_true',help='Also restore raw .bin files under data/raw_bin/<dataset>')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); cat=json.loads((ROOT/'data/catalog.json').read_text(encoding='utf-8'))
    selected=[d for d in cat['datasets'] if a.dataset in ['all',d['name']]]
    labels={d['name'] for d in selected}
    for item in cat['artifacts']:
        if item['dataset'] not in labels or (item['kind']=='raw_zip' and not a.raw): continue
        f=ROOT/item['path']
        if not f.is_file() or f.stat().st_size!=item['bytes'] or sha(f)!=item['sha256']:
            raise RuntimeError('Missing/corrupt shard or Git LFS pointer: '+item['path'])
    if a.verify_only: print('Selected shards verified'); return
    for d in selected:
        for file in d['processed_files']:
            dest=ROOT/d['processed_folder']/file['filename']
            if dest.exists():
                if sha(dest)==file['sha256']: continue
                raise FileExistsError(dest)
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp=Path(str(dest)+'.restoring')
            with temp.open('xb') as out:
                for part in file['parts']:
                    with (ROOT/part).open('rb') as src:
                        for block in iter(lambda:src.read(8*1024*1024),b''): out.write(block)
            if sha(temp)!=file['sha256']: raise RuntimeError('Reassembly checksum mismatch')
            temp.rename(dest)
        if a.raw:
            target=ROOT/'data'/'raw_bin'/d['name']; target.mkdir(parents=True,exist_ok=True)
            for item in cat['artifacts']:
                if item['dataset']!=d['name'] or item['kind']!='raw_zip': continue
                with zipfile.ZipFile(ROOT/item['path']) as z:
                    for name in z.namelist():
                        if Path(name).name!=name or not name.endswith('.bin'): raise ValueError('Unsafe archive member')
                        dest=target/name
                        if dest.exists(): raise FileExistsError(dest)
                        with z.open(name) as src,dest.open('xb') as out:
                            for b in iter(lambda:src.read(1024*1024),b''): out.write(b)
        print(d['name'],'restored')
if __name__=='__main__': main()
