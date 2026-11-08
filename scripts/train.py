"""Seed-specific Stage 1 -> warm-start -> guidance -> frozen-test pipeline."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPECS={name:(name,f'configs/{name}/warmup.json',f'configs/{name}/guidance.json') for name in ['NR-I','NR-II','FR']}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=SPECS,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,123,456,789,1024])
    p.add_argument('--device',default='cuda')
    p.add_argument('--output-root',type=Path,default=Path('results/paper'))
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); dataset,warm,guidance=SPECS[a.dataset]
    def run(cmd):
        command=[sys.executable,'-B',*map(str,cmd)]
        print(subprocess.list2cmdline(command),flush=True)
        if not a.dry_run: subprocess.run(command,cwd=ROOT,check=True)
    for seed in a.seeds:
        out=a.output_root/a.dataset/f'seed{seed}'
        if not a.dry_run and (ROOT/out).exists(): raise FileExistsError('Choose a new output root')
        s1=out/'stage1/model.stage1.best.pt'
        run(['model.py','--mode','stage1','--train-npz',f'data/ready/{dataset}/train_sequence.npz','--val-npz',f'data/ready/{dataset}/val_sequence.npz','--stats-npz',f'data/ready/{dataset}/normalization_stats.npz','--seed',seed,'--stage1-epochs',10,'--batch-size',16,'--lr',.001,'--save-path',out/'stage1/model.pt','--device',a.device])
        run(['scripts/train_stage2.py','--dataset',dataset,'--seed',seed,'--config',warm,'--stage1-checkpoint',s1,'--output-dir',out/'warmstart','--device',a.device])
        cfg=json.loads((ROOT/warm).read_text()); horizon=max(cfg['stage2']['rollout_curriculum'])
        ckpt=out/f"warmstart/{dataset}_seed{seed}_h{horizon}_sel-{cfg['selection']['criterion']}.best.pt"
        run(['scripts/train_stage2.py','--dataset',dataset,'--seed',seed,'--config',guidance,'--stage1-checkpoint',s1,'--stage2-init-checkpoint',ckpt,'--output-dir',out/'selection','--device',a.device])
        run(['scripts/evaluate.py','--source-run',out/'selection','--output-dir',out/'frozen_test','--device',a.device])
if __name__=='__main__': main()
