"""Supplementary descriptive pixel diagnostics; no gates or inference changes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.offline_obstacle_protocol import red_mask,verify_protocol
from research.isaac_final_common import write_json


def summarize(manifest):
    records=[];images=[]
    for p,a in zip(manifest['cases'][::2],manifest['cases'][1::2]):
        pa=np.array(Image.open(p['history'][-1]['path'])).astype(float)
        aa=np.array(Image.open(a['history'][-1]['path'])).astype(float)
        luminance=(aa-pa)@np.array([.2126,.7152,.0722])
        ground=np.indices((480,640))[0]>=240
        candidate=(luminance>5)&ground&~red_mask(pa)
        records.append(dict(pose_id=p['pose_id'],shadow_associated_darkening_candidate_pixels=int(candidate.sum()),
            mean_absent_minus_present_luminance_in_candidate=float(luminance[candidate].mean()),
            max_absent_minus_present_luminance_in_candidate=float(luminance[candidate].max())))
        vals=[]
        for c in (p,a):
            imgs=[np.array(Image.open(f['path'])).astype(float) for f in c['history']]
            vals.append([float(np.abs(im-imgs[0]).mean()) for im in imgs[1:]])
        images.append((pa,luminance,candidate,vals))
    result=dict(analysis_type='Post-inference descriptive input analysis; no gate, screen, or experiment condition changed',
        method='Outside red-dominant mask, image rows >=240, Rec.709 luminance ABSENT-PRESENT >5/255; shadow-associated darkening proxy, not semantic segmentation',pairs=records)
    return result,images


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();_,_,manifest,_=verify_protocol(ROOT/'research/frozen-edge-obstacle-offline.json')
    result,images=summarize(manifest);args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'input-difference-details.json',result)
    fig,axes=plt.subplots(4,3,figsize=(15,13),constrained_layout=True)
    for i,(record,(pa,luminance,candidate,vals)) in enumerate(zip(result['pairs'],images)):
        ax=axes[i]
        ax[0].imshow(luminance,cmap='coolwarm',vmin=-25,vmax=25)
        ax[0].set_title(record['pose_id']+' | luminance ABSENT - PRESENT')
        ax[1].imshow(pa.astype('uint8'))
        ax[1].imshow(np.ma.masked_where(~candidate,candidate),cmap='autumn',alpha=.55)
        ax[1].set_title('Ground darkening >5 outside red object (proxy)')
        ax[2].plot(range(1,6),vals[0],'o-',label='Present');ax[2].plot(range(1,6),vals[1],'o-',label='Absent')
        ax[2].set(xlabel='Independent capture index',ylabel='Mean |RGB difference| (0-255)',title='Stationary variation vs first capture')
        ax[2].legend()
        for j in (0,1):ax[j].axis('off')
    fig.suptitle('Descriptive input diagnostics only; shadow proxy also includes indirect lighting, edge and renderer effects')
    fig.savefig(args.output/'input-noise-and-shadow.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
