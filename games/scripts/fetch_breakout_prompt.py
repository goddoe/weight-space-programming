"""Fetch the exact upstream prompt, including its intact third-party notices."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]
COMMIT='4d09ba6d6a7e891dd8063974dfe81b3d78e3494f'
REL='exp5_write_from_scratch/runs/breakout/reply/prompt.txt'
EXPECTED='766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876'

def main():
    data=urlopen(f'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{COMMIT}/{REL}',timeout=60).read()
    if hashlib.sha256(data).hexdigest()!=EXPECTED:raise ValueError('Upstream prompt digest mismatch')
    for relative in ['lm_understand-logs/'+REL,'games/runs/breakout/prompt.txt']:
        path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print('Exact upstream Breakout prompt fetched and SHA256 verified.')

if __name__=='__main__':main()
