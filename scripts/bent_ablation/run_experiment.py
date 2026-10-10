"""Two fixed-seed bent runs. No test-set reads and no edits to upstream source."""
import argparse,hashlib,json,os,random,subprocess,sys,runpy
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--manifest',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--weight',required=True,type=float,choices=[0.,.1]);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
up=Path(a.upstream).resolve();out=Path(a.out).resolve();root=Path(a.data).resolve();manifest=Path(a.manifest).resolve()
pin='5a8099204bc0858da314f8e0b5e8b6770c298e98'
assert subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()==pin
assert not subprocess.check_output(['git','-C',str(up),'diff','HEAD','--'],text=True).strip()
import numpy as np,torch
from PIL import Image
import albumentations
assert albumentations.__version__=='1.3.1'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='b70c8ccc3668e89c2d2b17d527b9ab5e53c3d9646190abb44ae748c290db7bab'
m=json.loads(manifest.read_text());normal=[r for r in m['train_normal'] if r['category']=='metal_nut'];support=[r for r in m['support'] if r['category']=='metal_nut' and r['defect_type']=='bent']
assert len(normal)==220 and len(support)==5
assert not ({r['image_path'] for r in normal+support}&{r['image_path'] for r in m['heldout']})
for r in normal+support:
 for key in ['image']+(['mask'] if r in support else []):
  assert hashlib.sha256((root/r[key+'_path']).read_bytes()).hexdigest()==r[key+'_sha256']
source=(up/'train.py').read_text();assert source.count('cudnn.benchmark = True')==1
# Upstream enables benchmark unconditionally; override only this line in memory for both arms.
source=source.replace('cudnn.benchmark = True','cudnn.benchmark = False')
sys.path.insert(0,str(up))
runpy.run_path(str(up/'train.py'),run_name='dependency_preflight')
if a.dry_run:
 print('PREFLIGHT_OK: pinned source, 220 normals, 5 paired supports, hashes, compatible augmentation');sys.exit(0)
assert torch.cuda.is_available(),'GPU allocation required'
out.mkdir(parents=True,exist_ok=False);stage=out/'data'/'bent'
for d in ['A','B','mask']:(stage/'train'/d).mkdir(parents=True)
for i,r in enumerate(normal):Image.open(root/r['image_path']).convert('RGB').save(stage/'train/A'/f'{i:03}.png')
for i,r in enumerate(support):
 Image.open(root/r['image_path']).convert('RGB').save(stage/'train/B'/f'{i:03}.png')
 mask=Image.open(root/r['mask_path']);assert set(np.unique(np.array(mask)))<={0,255} and np.any(np.array(mask));mask.save(stage/'train/mask'/f'{i:03}_mask.png')
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True,warn_only=True)
args=['train.py','--cuda','--dataset','bent','--dataroot',str(out/'data'),'--weightsf',str(out/'weights'),'--outf',str(out/'images'),'--epochs','50','--decay_epochs','25','--save_freq','25','--lambda_background',str(a.weight)]
(out/'provenance.json').write_text(json.dumps(dict(seed=42,epochs=50,decay_epochs=25,lambda_background=a.weight,upstream=pin,normal=[r['image_path'] for r in normal],support=support,argv=args,source_override='cudnn.benchmark False in both arms',gpu=torch.cuda.get_device_name(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),indent=2))
(out/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
os.chdir(up);sys.path.insert(0,str(up));sys.argv=args
exec(compile(source,str(up/'train.py'),'exec'),{'__name__':'__main__','__file__':str(up/'train.py')})
(out/'TRAINING_COMPLETE').write_text('50 epochs\n')
