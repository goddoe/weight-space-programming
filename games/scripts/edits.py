"""Parse the experiments' numeric edits language without executing arbitrary code."""
import ast
import re
import numpy as np

BREAKOUT_NAMES={'W1':'actor/dense1/kernel','b1':'actor/dense1/bias','W2':'actor/dense2/kernel','b2':'actor/dense2/bias','W3':'actor/logits/kernel','b3':'actor/logits/bias'}


def breakout_zero():
    return {k:np.zeros(shape,dtype=np.float32) for k,shape in {'W1':(600,32),'b1':(32,),'W2':(32,32),'b2':(32,),'W3':(32,6),'b3':(6,)}.items()}


def car_from_prompt(text):
    pattern=re.compile(r'^(embed|actor\d+|actor_head)\.(W|b) \(([\d x]+)\), scale ([\deE+.-]+):[^\n]*\n',re.M)
    matches=list(pattern.finditer(text));out={}
    for i,m in enumerate(matches):
        shape=tuple(int(s.strip()) for s in m[3].split('x'))
        body=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        body=re.sub(r'^\d+:\s*','',body,flags=re.M)
        numbers=np.fromstring(body,sep=' ',dtype=np.float64)
        if numbers.size!=np.prod(shape):raise ValueError(f'Wrong tensor size {m[0]}: {numbers.size}')
        out[f'{m[1]}.{m[2]}']=(numbers*float(m[4])).reshape(shape)
    if len(out)!=14:raise ValueError('Expected 14 actor tensors')
    return out


def name(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name):return node.value.id+'.'+node.attr
    raise ValueError('Invalid tensor name')


def integer(node):
    if node is None:return None
    v=ast.literal_eval(node)
    if not isinstance(v,int) or isinstance(v,bool):raise ValueError('Index must be integer')
    return v


def index(node):
    if isinstance(node,ast.Tuple):return tuple(index(x) for x in node.elts)
    if isinstance(node,ast.Slice):return slice(integer(node.lower),integer(node.upper),integer(node.step))
    return integer(node)


def apply(answer,original):
    blocks=re.findall(r'^```edits\s*\n(.*?)^```',answer,re.M|re.S)
    if not blocks:raise ValueError('No edits block')
    statements=[];pending='';depth=0
    for line in blocks[-1].splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        pending+=' '+line;depth+=line.count('[')-line.count(']')
        if depth<=0:statements.append(pending.strip());pending='';depth=0
    if pending:statements.append(pending.strip())
    tensors={k:v.copy() for k,v in original.items()};errors=[];applied=0
    for i,statement in enumerate(statements,1):
        try:
            if statement.startswith('ZERO '):
                tensors[statement[5:].strip()][:]=0;applied+=1;continue
            tree=ast.parse(statement)
            if len(tree.body)!=1:raise ValueError('One statement per line required')
            node=tree.body[0]
            if isinstance(node,ast.Assign) and len(node.targets)==1:target=node.targets[0];op='='
            elif isinstance(node,ast.AugAssign) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
                target=node.target;op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*'}[type(node.op)]
            else:raise ValueError('Only numeric assignment and +=/-=/*= allowed')
            if not isinstance(target,ast.Subscript):raise ValueError('Expected indexed tensor')
            key=name(target.value);ix=index(target.slice)
            value=np.asarray(ast.literal_eval(node.value))
            if value.dtype.kind not in 'iuf' or not np.isfinite(value).all() or np.max(np.abs(value),initial=0)>1e6:raise ValueError('Non-numeric, nonfinite or oversized value')
            dst=tensors[key]
            if op=='=':dst[ix]=value
            elif op=='+':dst[ix]+=value
            elif op=='-':dst[ix]-=value
            else:dst[ix]*=value
            if not np.isfinite(dst).all():raise ValueError('Nonfinite resulting tensor')
            applied+=1
        except Exception as exc:errors.append({'statement':i,'text':statement,'error':str(exc)})
    report={'statements':len(statements),'applied':applied,'errors':errors,'changed':sum(int(np.count_nonzero(tensors[k]!=v)) for k,v in original.items()),'nonzero':sum(int(np.count_nonzero(v)) for v in tensors.values())}
    return tensors,report
