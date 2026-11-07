"""MinAtar 1.0.15 evaluation and faithful state-rendered MP4 capture."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from minatar import Environment
from PIL import Image,ImageDraw,ImageFont
from edits import BREAKOUT_NAMES,breakout_zero,apply

ROOT=Path(__file__).resolve().parents[2]


def weights(path):
    data=np.load(path)
    return {k:np.asarray(data[v] if v in data else data[k],dtype=np.float32) for k,v in BREAKOUT_NAMES.items()}


def obs(env):
    raw=env.state();x=np.zeros((10,10,6),dtype=np.float32)
    x[:,:,0]=raw[:,:,0];x[:,:,2]=raw[:,:,3];x[:,:,3]=raw[:,:,1];x[:,:,4]=raw[:,:,2]
    return x.reshape(-1)


def policy(x,w):
    h=np.tanh(x@w['W1']+w['b1']);h=np.tanh(h@w['W2']+w['b2']);z=h@w['W3']+w['b3']
    p=np.exp(z-z.max(axis=-1,keepdims=True));return p/p.sum(axis=-1,keepdims=True)


def evaluate(w,n=1000,seed=0,batch=32,output=None):
    started=time.monotonic();records=[]
    for start in range(0,n,batch):
        count=min(batch,n-start);envs=[];rngs=[]
        for i in range(count):
            e=Environment('breakout',sticky_action_prob=.1);e.seed(seed+start+i);e.reset();envs.append(e)
            rngs.append(np.random.default_rng(seed+1000000+start+i))
        scores=np.zeros(count);lengths=np.zeros(count,dtype=int);active=list(range(count))
        for step in range(10000):
            if not active:break
            probs=policy(np.stack([obs(envs[i]) for i in active]),w)
            keep=[]
            for j,i in enumerate(active):
                a=int(np.searchsorted(np.cumsum(probs[j],dtype=np.float64),rngs[i].random(),side='right'));a=min(a,5)
                reward,done=envs[i].act(a);scores[i]+=reward;lengths[i]+=1
                if not done:keep.append(i)
            active=keep
        records.extend({'episode':start+i,'seed':seed+start+i,'score':float(scores[i]),'steps':int(lengths[i])} for i in range(count))
        summary={'episodes':len(records),'mean':float(np.mean([r['score'] for r in records])),
                 'standard_error':float(np.std([r['score'] for r in records],ddof=1)/np.sqrt(len(records))) if len(records)>1 else None,
                 'mean_steps':float(np.mean([r['steps'] for r in records])),'wall_s':time.monotonic()-started,
                 'minatar_version':'1.0.15','sticky':.1,'max_steps':10000,'actions':'sampled softmax','seed':seed,'batch':batch}
        if output:
            output.mkdir(parents=True,exist_ok=True);(output/'scores.json').write_text(json.dumps(summary,indent=2));(output/'episodes.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(summary),flush=True)
        if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk below 5 GiB')
    return summary


def font(size):
    return ImageFont.load_default(size=size)


def frame(env,score,step,title):
    im=Image.new('RGB',(960,1080),'#0b1220');d=ImageDraw.Draw(im)
    d.text((48,35),'WEIGHT-SPACE PROGRAMMING',font=font(23),fill='#8fa7c7')
    d.text((48,80),title,font=font(35),fill='white')
    d.text((48,137),f'Score {int(score):3d}     Step {step:5d}     20,486 weights',font=font(24),fill='#b8c7db')
    s=env.state();cell=80;ox=80;oy=220
    for y in range(10):
        for x in range(10):
            box=(ox+x*cell+2,oy+y*cell+2,ox+(x+1)*cell-2,oy+(y+1)*cell-2)
            d.rounded_rectangle(box,7,fill='#162239')
            if s[y,x,3]:d.rounded_rectangle((box[0]+3,box[1]+20,box[2]-3,box[3]-20),7,fill=['#fa9967','#e9c55c','#5cd5b1'][max(0,min(2,y-1))])
            if s[y,x,2]:d.ellipse((box[0]+27,box[1]+27,box[2]-27,box[3]-27),fill='#56647c')
            if s[y,x,1]:d.ellipse((box[0]+18,box[1]+18,box[2]-18,box[3]-18),fill='white')
            if s[y,x,0]:d.rounded_rectangle((box[0]+2,box[1]+45,box[2]-2,box[3]-5),8,fill='#6cacff')
    d.text((48,1030),'Actual MinAtar states | sampled policy | sticky actions 0.1',font=font(20),fill='#8fa7c7')
    return im


def video(w,path,seed=0,steps=1800,title='Astra writes a Breakout policy'):
    env=Environment('breakout',sticky_action_prob=.1);env.seed(seed);env.reset();rng=np.random.default_rng(seed+1000000)
    path.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    score=0;trace=[]
    with path.with_suffix('.ffmpeg.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for step in range(steps):
            proc.stdin.write(frame(env,score,step,title).tobytes())
            probs=policy(obs(env)[None,:],w)[0];action=min(int(np.searchsorted(np.cumsum(probs,dtype=np.float64),rng.random(),side='right')),5)
            reward,done=env.act(action);score+=reward;trace.append({'step':step,'action':action,'reward':reward,'score':score})
            if done:break
        ending=frame(env,score,len(trace),title)
        for _ in range(60):proc.stdin.write(ending.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError('ffmpeg failed')
    path.with_suffix('.json').write_text(json.dumps({'seed':seed,'score':score,'steps':len(trace),'ended':done,'clip_cap':steps,'fps':30,'trace':trace},indent=2))
    print({'video':str(path),'score':score,'steps':len(trace),'ended':done},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path);p.add_argument('--answer',type=Path);p.add_argument('--episodes',type=int,default=1000);p.add_argument('--out',type=Path,default=ROOT/'games/runs/breakout/eval');p.add_argument('--video',type=Path);p.add_argument('--video-only',action='store_true');a=p.parse_args()
    if a.answer:
        w,r=apply(a.answer.read_text(),breakout_zero());a.out.mkdir(parents=True,exist_ok=True);(a.out/'apply.json').write_text(json.dumps(r,indent=2));np.savez(a.out/'actor.npz',**{BREAKOUT_NAMES[k]:v for k,v in w.items()})
    else:w=weights(a.weights)
    if not a.video_only:evaluate(w,a.episodes,output=a.out)
    if a.video:video(w,a.video)


if __name__=='__main__':main()
