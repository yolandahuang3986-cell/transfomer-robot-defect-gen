"""Fixed normal-training preview after each arm; inference always eval mode."""
import argparse,functools,json,sys,time,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--run',required=True);p.add_argument('--manifest',required=True);p.add_argument('--data',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(a.upstream).resolve()))
from src.models_mask import ResNetGenerator9
torch.set_num_threads(2);run=Path(a.run);out=run/'preview';out.mkdir(exist_ok=False)
m=json.loads(Path(a.manifest).read_text());normal=sorted([r for r in m['train_normal'] if r['category']=='metal_nut'],key=lambda r:r['image_path'])[:5];summary={}
for epoch in [25,50]:
 checkpoint=run/f'weights/bent/netG_A2B_epoch_{epoch}.pth';net=ResNetGenerator9(3,3,64,norm_layer=functools.partial(torch.nn.BatchNorm2d,affine=True,track_running_stats=True),n_blocks=9)
 net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);net.eval();canvas=Image.new('RGB',(768,280*len(normal)),'white');draw=ImageDraw.Draw(canvas);records=[];probs=[]
 for i,r in enumerate(normal):
  im=Image.open(Path(a.data)/r['image_path']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC);x=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()[None]/127.5-1
  start=time.perf_counter()
  with torch.no_grad():rgb,prob,_=net(x)
  assert torch.isfinite(rgb).all() and torch.isfinite(prob).all()
  rgb=((rgb[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy();pm=prob[0].numpy();probs.append(pm);mask=(pm>.5).astype('uint8')*255
  Image.fromarray(rgb).save(out/f'epoch{epoch}_{i}_image.png');Image.fromarray(mask).save(out/f'epoch{epoch}_{i}_mask.png')
  records.append(dict(source_image_id=r['image_path'],generation_seconds=time.perf_counter()-start,area05=float((pm>.5).mean()),area02=float((pm>.2).mean()),max_probability=float(pm.max())))
  for j,img in enumerate([im,Image.fromarray(rgb),Image.fromarray(mask).convert('RGB')]):canvas.paste(img,(j*256,i*280+24))
  draw.text((4,i*280+4),f'epoch {epoch}: normal / generated / mask >0.5',fill='black')
 canvas.save(out/f'epoch{epoch}_contact.jpg');np.savez_compressed(out/f'epoch{epoch}_probabilities.npz',probabilities=np.stack(probs));summary[str(epoch)]=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),records=records)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('PREVIEW_COMPLETE')
