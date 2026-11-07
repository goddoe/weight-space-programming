"""Three explicitly authorized independent fresh responses; no evaluation feedback."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PY=ROOT/'.venv-games/bin/python'
S=ROOT/'games/scripts'
env=os.environ.copy()
env.update(JAX_PLATFORMS='cpu',JAX_COMPILATION_CACHE_DIR=str(ROOT/'.cache/jax'),MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')

def run(args,log):
    if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk below 5 GiB')
    with log.open('wb') as f:subprocess.run([str(PY),*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    results=[]
    for i in range(1,4):
        name=f'car_repeat{i}';out=ROOT/'games/runs'/name;out.mkdir(exist_ok=True)
        run([S/'ask_astra.py','car','--run-name',name],out/'driver.log')
        metadata=json.loads((out/'run.json').read_text())
        if metadata['status']!='complete' or not metadata['tool_free']:raise RuntimeError('Invalid Astra response; halt for review')
        run([S/'car.py','--answer',out/'answer.md','--episodes','256','--seed','0','--out',out/'search','--trajectory'],out/'search.log')
        run([S/'car.py','--weights',out/'search/actor.npz','--episodes','1024','--seed','100000','--out',out/'test'],out/'test.log')
        search=json.loads((out/'search/scores.json').read_text());test=json.loads((out/'test/scores.json').read_text());edits=json.loads((out/'search/apply.json').read_text())
        record={'run':name,'search':search,'test':test,'edits':edits,'wall_s':metadata['wall_s'],'usage':metadata['usage']}
        results.append(record);(ROOT/'games/runs/car_repeats_summary.json').write_text(json.dumps(results,indent=2))
        message=f'자동차 추가 {i}/3 완료: search {round(search["success"]*256)}/256, test {round(test["success"]*1024)}/1024 성공. 편집 {edits["applied"]}/{edits["statements"]}, 변경 가중치 {edits["changed"]}. Astra {metadata["wall_s"]:.1f}초, 출력 {metadata["usage"][0]["output_tokens"]}토큰. 새 대화·원문 프롬프트·도구 off, 평가 피드백 없음.'
        print(message,flush=True)
    print('All three independent responses evaluated.',flush=True)

if __name__=='__main__':main()
