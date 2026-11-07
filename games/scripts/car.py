"""Kinetix ramp policy evaluation on CPU, plus genuine state trajectory capture."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import jax
import jax.numpy as jnp
from kinetix.environment import make_kinetix_env,ActionType,ObservationType
from kinetix.util import load_from_json_file
from edits import car_from_prompt,apply

ROOT=Path(__file__).resolve().parents[2]


def environment():
    level,static,params=load_from_json_file('l/h2_car_ramp.json')
    static=static.replace(frame_skip=2);params=params.replace(max_timesteps=256,dense_reward_scale=.2)
    env=make_kinetix_env(action_type=ActionType.MULTI_DISCRETE,observation_type=ObservationType.SYMBOLIC_FLAT,reset_fn=lambda rng:level,env_params=params,static_env_params=static,auto_reset=False)
    x,state=env.reset(jax.random.PRNGKey(0),params)
    assert x.shape==(679,)
    return env,params,state


def actor(obs,w):
    x=jnp.clip(jnp.nan_to_num(obs,nan=0,posinf=0,neginf=0),-10,10)
    h=jax.nn.relu(x@w['embed.W']+w['embed.b'])
    for i in range(5):h=jnp.tanh(h@w[f'actor{i}.W']+w[f'actor{i}.b'])
    return h@w['actor_head.W']+w['actor_head.b']


def make_rollout(capture=False):
    env,params,initial=environment()
    def rollout(weights,seed):
        def step(carry,_):
            state,key,finished,outcome,total,length=carry
            keys=jax.random.split(key,8)
            z=actor(env.get_obs(state),weights)
            action=jnp.stack([jax.random.categorical(keys[i+1],z[a:b]) for i,(a,b) in enumerate([(0,3),(3,6),(6,9),(9,12),(12,14),(14,16)])])
            _,new,reward,done,info=env.step(keys[7],state,action,params)
            invalid=~jax.tree.reduce(jnp.logical_and,jax.tree.map(lambda x:jnp.isfinite(x).all(),new),True)
            done=done|invalid
            code=jnp.where(invalid,3,jnp.where(info['GoalR'],1,jnp.where(new.timestep<256,2,0)))
            outcome=jnp.where((~finished)&done,code,outcome)
            total=total+jnp.where(finished,0,reward);length=length+(~finished).astype(jnp.int32)
            state=jax.tree.map(lambda a,b:jnp.where(finished,a,b),state,new)
            finished=finished|done
            trace={'polygon_position':state.polygon.position,'polygon_rotation':state.polygon.rotation,
                   'circle_position':state.circle.position,'circle_rotation':state.circle.rotation,
                   'action':action,'outcome':outcome,'done':finished} if capture else None
            return (state,keys[0],finished,outcome,total,length),trace
        (_,_,_,outcome,total,length),trace=jax.lax.scan(step,(initial,jax.random.PRNGKey(seed),jnp.array(False),jnp.array(0),jnp.array(0.),jnp.array(0)),None,length=256)
        return {'outcome':outcome,'return':total,'length':length},trace
    return rollout,initial


def evaluate(weights,n,seed,out):
    fn,_=make_rollout();batched=jax.jit(jax.vmap(fn,in_axes=(None,0)));records=[];started=time.monotonic()
    w={k:jnp.asarray(v,dtype=jnp.float32) for k,v in weights.items()}
    for start in range(0,n,32):
        if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk below 5 GiB')
        count=min(32,n-start);res,_=batched(w,jnp.arange(seed+start,seed+start+32,dtype=jnp.int32));res=jax.device_get(res)
        records.extend({'seed':seed+start+i,'outcome':int(res['outcome'][i]),'return':float(res['return'][i]),'length':int(res['length'][i])} for i in range(count))
        summary={'episodes':len(records),'success':sum(r['outcome']==1 for r in records)/len(records),'red':sum(r['outcome']==2 for r in records),
                 'invalid':sum(r['outcome']==3 for r in records),'timeout':sum(r['outcome']==0 for r in records),
                 'mean_return':float(np.mean([r['return'] for r in records])),'mean_length':float(np.mean([r['length'] for r in records])),
                 'wall_s':time.monotonic()-started,'seed_start':seed,'frame_skip':2,'dense_reward_scale':.2,'jax':jax.__version__,'platform':'cpu'}
        out.mkdir(parents=True,exist_ok=True);(out/'scores.json').write_text(json.dumps(summary,indent=2));(out/'episodes.json').write_text(json.dumps(records,indent=2));print(json.dumps(summary),flush=True)
    return summary


def trajectory(weights,path,seed=100000):
    fn,state=make_rollout(capture=True);res,trace=jax.jit(fn)({k:jnp.asarray(v,dtype=jnp.float32) for k,v in weights.items()},seed)
    res,trace=jax.device_get((res,trace))
    static={'polygon_vertices':np.asarray(state.polygon.vertices),'polygon_n_vertices':np.asarray(state.polygon.n_vertices),'polygon_active':np.asarray(state.polygon.active),
            'polygon_roles':np.asarray(state.polygon_shape_roles),'circle_radius':np.asarray(state.circle.radius),'circle_active':np.asarray(state.circle.active),'circle_roles':np.asarray(state.circle_shape_roles)}
    np.savez_compressed(path,**trace,**static)
    path.with_suffix('.json').write_text(json.dumps({k:np.asarray(v).item() for k,v in res.items()}|{'seed':seed},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--answer',type=Path);p.add_argument('--weights',type=Path);p.add_argument('--base',action='store_true');p.add_argument('--episodes',type=int,default=1024);p.add_argument('--seed',type=int,default=100000);p.add_argument('--out',type=Path,required=True);p.add_argument('--trajectory',action='store_true');a=p.parse_args()
    base=car_from_prompt((ROOT/'lm_understand-logs/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt').read_text())
    if a.answer:
        w,report=apply(a.answer.read_text(),base);a.out.mkdir(parents=True,exist_ok=True);(a.out/'apply.json').write_text(json.dumps(report,indent=2));np.savez(a.out/'actor.npz',**w)
    elif a.weights:w=dict(np.load(a.weights))
    else:w=base
    evaluate(w,a.episodes,a.seed,a.out)
    if a.trajectory:trajectory(w,a.out/'trajectory.npz',a.seed)


if __name__=='__main__':main()
