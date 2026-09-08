"""Relative-path addendum runtime; base and scientific source bytes stay fixed."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
EXT_REL='SCI_Manuscript/experiments/research_v09_set_attention'
PLAN_SHA='8d21e13af116dc97556b37060daaf86afed5f7ba645a5bdae18e3d4ffb3b05a9'
BASE_SHA='157f84f2dab43b4f4070f0266232ac4b3f38588f5a5c887a411f61efa69432f6'
def io(p):
    s=str(Path(p).resolve())
    if os.name=='nt' and not s.startswith('\\\\?\\'):
        s='\\\\?\\UNC\\'+s[2:] if s.startswith('\\\\') else '\\\\?\\'+s
    return Path(s)
def read(p): return json.loads(io(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with io(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def local(r):
    p=(ROOT/r).resolve()
    if not p.is_relative_to(ROOT): raise ValueError('Addendum payload path escapes its root')
    return p
def write_report(p,obj):
    p=Path(p).resolve()
    if p.is_relative_to(ROOT): raise ValueError('Write actual execution receipts outside the sealed addendum')
    p.parent.mkdir(parents=True,exist_ok=True)
    with io(p).open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,io(path))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def initialize(base_override=None):
    manifest=read(ROOT/'addendum_manifest.json')
    assert manifest['status']=='SEALED_LOCAL_ATTENTION60_BUDGET_EVIDENCE_ADDENDUM'
    assert manifest['extension_plan_sha256']==PLAN_SHA and manifest['additional_checkpoints']==60
    for r,item in manifest['files'].items():
        p=io(local(r));assert p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],r
    dep=read(ROOT/'base_dependency.json')
    base=(ROOT/(base_override or dep['default_relative_directory'])).resolve()
    assert sha(base/'package_manifest.json')==dep['package_manifest_sha256']==BASE_SHA
    # The already-sealed runtime verifies the entire base payload and prepares
    # models/data from that copied base, never from historical original paths.
    lib=load_module('addendum_verified_base_lib',base/'portable_lib.py')
    package,rr,original,metadata,tensors,splits,original_registry=lib.initialize()
    for r,item in dep['required_files'].items():
        p=io(lib.local(r));assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],r
    plan_path=local(EXT_REL+'/plan_set_attention_60_v1.json')
    assert sha(plan_path)==PLAN_SHA
    plan=read(plan_path);mapping=read(ROOT/'relocation_map.json')
    assert mapping['original_extension_plan_sha256']==PLAN_SHA and mapping['original_plan_modified'] is False
    assert plan['original_plan_sha256']==package['original_plan_sha256']
    assert plan['original_source_sha256']==original['source_sha256']==rr.sources()
    assert plan['training_configuration']==original['training_configuration']
    assert plan['manifest_sha256']==original['manifest_sha256']
    assert plan['seeds']==[42,43,44,45,46] and list(plan['protocols'])==list(splits)
    assert len(mapping['fields'])==14
    for field,item in mapping['fields'].items():
        old=plan[field] if field in ['manifest_path','runs_root'] else plan['protocols'][field.split('.')[1]]['split_path']
        assert item['original']==old
        actual=lib.local(item['relative']) if item['owner']=='base' else local(item['relative'])
        if field=='manifest_path': assert sha(actual)==plan['manifest_sha256']
        elif field.startswith('protocols.'): assert sha(actual)==plan['protocols'][field.split('.')[1]]['split_sha256']
        else: assert io(actual).is_dir()
    for entry in mapping['source_files']:
        assert plan['extension_source_sha256'][entry['original']]==entry['sha256']==sha(local(entry['relative']))
    review=read(local(EXT_REL+'/independent_full_matrix_review.json'))
    aggregate=local(mapping['aggregate_directory'])
    assert review['numerical_review_passed'] is True
    assert review['aggregate_report_sha256']==sha(aggregate/'report.json')
    assert review['aggregate_run_integrity_sha256']==sha(aggregate/'run_integrity.json')
    assert review['extension_plan_sha256']==PLAN_SHA
    registry=read(ROOT/'checkpoint_registry.json')
    actual_keys={(r['protocol'],r['model'],r['seed']) for r in registry}
    expected={(protocol,'set_attention',seed) for protocol in plan['protocols'] for seed in plan['seeds']}
    assert actual_keys==expected and len(registry)==60
    for r in registry:
        expected_folder=local(mapping['fields']['runs_root']['relative'])/r['protocol']/'set_attention'/f"seed_{r['seed']}"
        assert local(r['relative_directory'])==expected_folder
    return manifest,lib,rr,plan,metadata,tensors,splits,registry,base

def check_run(record,plan,metadata):
    folder=local(record['relative_directory'])
    meta,result,status=[read(folder/name) for name in ['run_metadata.json','result.json','status.json']]
    assert status['status']=='COMPLETE_EXPLORATORY_EXTENSION'
    for key,value in [('protocol',record['protocol']),('mode','set_attention'),('seed',record['seed'])]:
        assert meta[key]==result[key]==value
    assert meta['plan_sha256']==PLAN_SHA
    assert meta['original_plan_sha256']==plan['original_plan_sha256']
    assert meta['original_source_sha256']==plan['original_source_sha256']
    assert meta['extension_source_sha256']==plan['extension_source_sha256']
    assert meta['dataset_sha256']==plan['manifest_sha256']
    assert meta['split_sha256']==plan['protocols'][record['protocol']]['split_sha256']
    assert meta['configuration']==plan['training_configuration']
    assert meta['parameters']==plan['parameters']==194273 and meta['architecture']==plan['architecture']
    for key in ['target_version','feature_version','feature_tensor_sha256','target_tensor_sha256']:
        assert meta[key]==metadata[key]
    assert sha(folder/'best.pt')==result['checkpoint_sha256']==record['checkpoint_sha256']
    return folder,meta,result
