"""Render recorded Kinetix physics states, never interpolated or scripted motion."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def font(n):
    return ImageFont.load_default(size=n)

def panel(im,t,k,left,label):
    draw=ImageDraw.Draw(im)
    draw.text((left+32,160),label,font=font(32),fill='white')
    # Same fixed world view for both policies; 5 world units per panel.
    scale=176
    scene=Image.new('RGB',(900,620),'#111e31');sd=ImageDraw.Draw(scene)
    def pixel(p):return (10+float(p[0])*scale,570-float(p[1])*scale)
    colors={0:'#8494aa',1:'#5bd6a0',2:'#5ba2ff',3:'#fa686f'}
    for i,active in enumerate(t['polygon_active']):
        if not active:continue
        a=t['polygon_rotation'][k,i];c,s=np.cos(a),np.sin(a)
        rot=np.array([[c,-s],[s,c]])
        verts=t['polygon_vertices'][i,:int(t['polygon_n_vertices'][i])]@rot.T+t['polygon_position'][k,i]
        sd.polygon([pixel(p) for p in verts],fill=colors.get(int(t['polygon_roles'][i]),'#8494aa'))
    for i,active in enumerate(t['circle_active']):
        if not active:continue
        x,y=pixel(t['circle_position'][k,i]);r=float(t['circle_radius'][i])*scale
        sd.ellipse((x-r,y-r,x+r,y+r),fill=colors.get(int(t['circle_roles'][i]),'#8494aa'),outline='#e6edf8',width=2)
        a=float(t['circle_rotation'][k,i]);sd.line((x,y,x+r*np.cos(a),y-r*np.sin(a)),fill='#22334a',width=3)
    im.paste(scene,(left+30,260))
    outcome=int(t['outcome'][k]);done=bool(t['done'][k])
    status={1:'GOAL REACHED',2:'RED CONTACT',3:'INVALID STATE'}.get(outcome,'TIME LIMIT' if done else 'RUNNING')
    draw.text((left+32,910),status,font=font(28),fill='#5bd6a0' if outcome==1 else '#b8c7db')

def render(before,after,out,after_label='AFTER: new Astra edits'):
    b=dict(np.load(before));a=dict(np.load(after));out.parent.mkdir(parents=True,exist_ok=True)
    def frame(k):
        im=Image.new('RGB',(1920,1080),'#0b1220');d=ImageDraw.Draw(im)
        d.text((48,35),'WEIGHT-SPACE PROGRAMMING',font=font(27),fill='#8fa7c7')
        d.text((48,85),'Can a weight edit fix the failing car?',font=font(44),fill='white')
        d.line((960,160,960,970),fill='#283b55',width=2)
        panel(im,b,k,0,'BEFORE: rounded PPO weights');panel(im,a,k,960,after_label)
        d.text((48,1010),f'Actual Kinetix states | step {k+1}/256 | sampled actions | completed episodes held still',font=font(25),fill='#8fa7c7')
        return im
    cmd=['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with out.with_suffix('.ffmpeg.log').open('wb') as log:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for k in list(range(256))+[255]*90:p.stdin.write(frame(k).tobytes())
        p.stdin.close()
        if p.wait():raise RuntimeError('ffmpeg failed')
    frame(60).save(out.with_suffix('.png'))
    out.with_suffix('.json').write_text(json.dumps({'before':str(before),'after':str(after),'after_label':after_label,'frames':346,'fps':30,'source':'actual recorded simulator states'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--after-label',default='AFTER: new Astra edits');a=p.parse_args();render(a.before,a.after,a.out,a.after_label)
