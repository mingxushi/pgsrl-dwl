"""Full closed-loop prediction using only the initial 12 spectrum/wind pairs."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_stage2 import build_model,make_rollout_config
from rollout import rollout
from model import load_normalization_stats
from utils import denormalize_wind
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--stage1-checkpoint',type=Path,required=True)
    p.add_argument('--initial-npz',type=Path,required=True,help='Normalized spectra[12,60,100], wind_speed[12,60]')
    p.add_argument('--stats-npz',type=Path,required=True)
    p.add_argument('--horizon',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.horizon<1: raise ValueError('horizon must be positive')
    if a.output.exists(): raise FileExistsError(a.output)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    cfg={k:v for k,v in checkpoint['config'].items() if k!='runtime'}
    if int(cfg['data']['input_steps'])!=12: raise ValueError('Paper context length is 12')
    with np.load(a.initial_npz,allow_pickle=False) as d:
        spectra=d['spectra'].astype(np.float32); wind=d['wind_speed'].astype(np.float32)
    if spectra.shape!=(12,60,100) or wind.shape!=(12,60): raise ValueError('Expected exactly 12 observations, 60 gates, 100 bins')
    sample={'spectrum_history':torch.from_numpy(spectra)}
    class Context:
        def __getitem__(self,index): return sample
    device=torch.device(a.device)
    model=build_model(cfg,Context(),a.stage1_checkpoint,device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        context=model.encode_history(torch.from_numpy(spectra).unsqueeze(0).to(device))
        result=rollout(model,context,torch.from_numpy(wind).unsqueeze(0).to(device),make_rollout_config(cfg,a.horizon,0.,train=False,delta_thresholds={}))
    stats=load_normalization_stats(str(a.stats_npz))
    arrays={name:denormalize_wind(result[key][0].cpu().numpy(),stats) for name,key in [('direct','direct'),('physics_decoded','raw_decode'),('guided','decode')]}
    arrays['gate']=result['guidance_gate'][0].cpu().numpy()
    for key in ['z_peak','z_width','z_noise','peak_pos']: arrays[key]=result[key][0].cpu().numpy()
    a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,**arrays)
    print('Saved',a.horizon,'free-running forecasts:',a.output)
if __name__=='__main__': main()
