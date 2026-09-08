"""Exercise the real generate entry on the actual incomplete matrix only."""
from pathlib import Path
from unittest.mock import patch
import argparse
import datetime
import json
import sys
import generate_extension_figures as gen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(gen.HERE/'incomplete_gate_probe_evidence_v2.json'))
    args=parser.parse_args()
    output=gen.HERE/'forbidden_incomplete_generation'
    require=lambda test:None if test else (_ for _ in ()).throw(AssertionError('Unexpected gate behavior'))
    require(not output.exists())
    attempts={'performance_files':0,'load_plot_data':0,'draw_figures':0,'write_tables':0,'mkdir':0}
    real_open=Path.open
    real_gate=gen.metadata_gate;captured={}
    def capture_gate(*args,**kwargs):
        plan,state=real_gate(*args,**kwargs);captured.update(state);return plan,state
    def guarded_open(path,*args,**kwargs):
        if path.name in ('result.json','summary.json','per_case_metrics.csv','test_predictions.npz','best.pt') or path.suffix in ('.npz','.pt'):
            attempts['performance_files']+=1;raise AssertionError('Gate attempted a quantitative payload read')
        return real_open(path,*args,**kwargs)
    def forbid(name):
        def blocked(*args,**kwargs):attempts[name]+=1;raise AssertionError('Gate attempted '+name)
        return blocked
    with patch.object(Path,'open',guarded_open),patch.object(Path,'mkdir',forbid('mkdir')),\
         patch.object(gen,'load_plot_data',forbid('load_plot_data')),patch.object(gen,'draw_figures',forbid('draw_figures')),\
         patch.object(gen,'write_tables',forbid('write_tables')),patch.object(gen,'metadata_gate',capture_gate):
        result=gen.main(['generate','--audit-dir',str(gen.EXT/'evaluation_final60'),
                        '--review',str(gen.EXT/'independent_full_matrix_review.json'),'--output',str(output)])
    require(result==2 and not output.exists() and not any(attempts.values()))
    require('matplotlib.pyplot' not in sys.modules)
    record={'status':'ACTUAL_INCOMPLETE_MATRIX_GENERATE_REFUSED_WITH_ZERO_QUANTITATIVE_READS',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'generator_sha256':gen.sha(Path(gen.__file__)),
        'plan_sha256':gen.PLAN_SHA,'entry':'generate with explicit audit/review/output arguments',
        'returncode':result,'actual_gate':captured,'attempted_calls':attempts,'output_directory_created':False,
        'plotting_backend_imported':False,'fake_data_used':False,'complete_generation_branch_tested':False,
        'full_result_review_passed':False,'probe_source_sha256':gen.sha(Path(__file__))}
    gen.write_json(gen.path_inside(args.output,gen.HERE),record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
