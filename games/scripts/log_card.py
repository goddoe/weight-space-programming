"""Export real invocation metadata and final-answer excerpts; not a screen recording."""
import argparse
import json
from pathlib import Path
import textwrap
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
def font(n):return ImageFont.load_default(size=n)
def render(game):
    folder=ROOT/'games/runs'/game;r=json.loads((folder/'run.json').read_text());u=r['usage'][0]
    im=Image.new('RGB',(1920,1080),'#0b1220');d=ImageDraw.Draw(im)
    lines=[('RECORDED CODEX RUN / '+game.upper(),'#8fa7c7'),('Exported log excerpt (not a screen recording)','#8fa7c7'),
        ('','white'),('model: gpt-6-astra    effort: '+r['effort'],'#f1f5fb'),
        ('tools: OFF    tool calls: 0    replies: '+str(r['item_types']['agent_message']),'#5bd6a0'),
        (f'wall time: {r["wall_s"]:.1f}s    output tokens: {u["output_tokens"]:,}','#f1f5fb'),
        (f'reasoning tokens: {u.get("reasoning_output_tokens",0):,}','#f1f5fb'),('','white'),('FINAL ANSWER EXCERPT','#8fa7c7')]
    answer=(folder/'answer.md').read_text().split('```')[0].strip().split('\n\n')[0]
    for line in textwrap.wrap(answer[:900],88):lines.append((line,'#f1f5fb'))
    y=55
    for text,color in lines[:23]:d.text((60,y),text,font=font(30),fill=color);y+=42
    out=ROOT/'games/deliverables';out.mkdir(exist_ok=True);im.save(out/f'{game}_execution_log.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game');render(p.parse_args().game)
