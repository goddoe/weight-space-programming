"""One-shot, tools-off Codex invocation with verbatim original experiment prompt."""
import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('game',choices=['breakout','car']);p.add_argument('--run-name');a=p.parse_args()
    run_name=a.run_name or a.game
    if not re.fullmatch(r'[a-z0-9_]+',run_name):raise ValueError('Invalid run name')
    if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk below 5 GiB')
    source=ROOT/('lm_understand-logs/exp5_write_from_scratch/runs/breakout/reply/prompt.txt' if a.game=='breakout' else 'lm_understand-logs/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt')
    out=ROOT/'games/runs'/run_name;out.mkdir(parents=True,exist_ok=True)
    if (out/'events.jsonl').exists():raise RuntimeError('Existing invocation: do not silently give Astra a second try')
    prompt=source.read_bytes();(out/'prompt.txt').write_bytes(prompt)
    work=ROOT/f'games/isolated_{run_name}';work.mkdir(exist_ok=True)
    effort='xhigh' if a.game=='breakout' else 'high'
    cmd=['codex','exec','--strict-config','--skip-git-repo-check','--json','-m','gpt-6-astra','-c',f'model_reasoning_effort="{effort}"','-c','model_context_window=872000','-c','model_auto_compact_token_limit=1000000','-o',str(out/'answer.md')]
    if a.game=='breakout':cmd+=['-c','model_reasoning_summary="detailed"']
    cmd+=['-']
    codex_home=ROOT/'.cache/codex-games';codex_home.mkdir(parents=True,exist_ok=True)
    instructions=codex_home/'instructions_neutral.md'
    instructions.write_text((ROOT/'config/instructions_neutral.md').read_text())
    config=(ROOT/'config/codex_config.toml').read_text().replace('<codex_home>',codex_home.as_posix())
    (codex_home/'config.toml').write_text(config)
    env=os.environ.copy();env['CODEX_HOME']=str(codex_home)
    env['UV_CACHE_DIR']=str(ROOT/'.cache/uv');env['HF_HOME']=str(ROOT/'.cache/hf')
    for key in ['CODEX_THREAD_ID','CODEX_SESSION_ID']:env.pop(key,None)
    start=time.time();record={'model':'gpt-6-astra','effort':effort,'game':a.game,'run_name':run_name,'cmd':cmd,'prompt_sha256':hashlib.sha256(prompt).hexdigest(),'started_at':start,'status':'running','tools_config_source':'lm_understand-logs/harness/codex_config.toml'}
    (out/'run.json').write_text(json.dumps(record,indent=2))
    with (out/'events.jsonl').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
        proc=subprocess.run(cmd,input=prompt,cwd=work,env=env,stdout=stdout,stderr=stderr)
    types=collections.Counter();usage=[];bad=[];complete=False
    for line in (out/'events.jsonl').read_text().splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        item=event.get('item',{});kind=item.get('type')
        if kind:
            types[kind]+=1
            if kind not in {'reasoning','agent_message','error'}:bad.append(kind)
        if event.get('type')=='turn.completed':complete=True;usage.append(event.get('usage',{}))
    record.update(status='complete' if proc.returncode==0 and complete and not bad else 'failed',returncode=proc.returncode,wall_s=time.time()-start,item_types=dict(types),tool_free=not bad,unexpected_items=bad,turn_completed=complete,usage=usage)
    (out/'run.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
