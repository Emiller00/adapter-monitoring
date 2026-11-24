"""Recover 011 score rows using the original research project and artifacts."""
import argparse
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from glnet_trace_monitoring.research_weights import OUTPUT, _lora_pairs, config, load_jsonl
from glnet_trace_monitoring.models import AdapterGLNet, PairSpec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", help="Path for recovered score rows in JSON format")
args = parser.parse_args()

stage = OUTPUT / '011'
population = json.loads((stage/'population.json').read_text())
report = json.loads((stage/'results/summary.json').read_text())
checkpoint = torch.load(stage/'models/adapter_glnet.pt', map_location='cpu', weights_only=True)
net = AdapterGLNet([PairSpec(*spec) for spec in checkpoint['specs']], hidden_rows=4,
                   pair_depth=2, token_width=32, head_width=32, dropout=.1).eval()
net.load_state_dict(checkpoint['state'])
rows = []
with torch.inference_mode():
    for item in population:
        block = item['block']
        if block < 12:
            continue
        pairs, _ = _lora_pairs(OUTPUT/item['adapter'])
        inputs = {name:(u.unsqueeze(0),v.unsqueeze(0)) for name,(u,v) in pairs.items()}
        weight = float(net(inputs, torch.ones((1,next(iter(pairs.values()))[0].shape[0]),dtype=torch.bool)).sigmoid()[0])
        for record in load_jsonl(OUTPUT/item['records']):
            label = 'violation' if record['task_violation'] else 'compliant' if record['valid_response'] else 'invalid'
            rows.append(dict(id=item['run_id']+'/'+record['row_id'], block=block, label=label,
                features=[float(record['trigger_present']), record['monitors']['before']['semantic_disclosure']['score'],weight],
                scores={}))
fit = [r for r in rows if r['block'] < 15]
scored = [r for r in rows if r['block'] >= 15]
for name, width in [('trace_only',2), ('trace_plus_weight',3)]:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config()['seed']))
    model.fit(np.array([r['features'][:width] for r in fit]),[r['label']=='violation' for r in fit])
    for r, score in zip(scored, model.predict_proba(np.array([r['features'][:width] for r in scored]))[:,1]):
        r['scores'][name] = float(score)
    cut = report['execution_prediction'][name]['threshold']
    print(name,[(label,sum(r['scores'][name]>=cut for r in scored if r['block']>=18 and r['label']==label)) for label in ('violation','compliant')])
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(scored,f)
