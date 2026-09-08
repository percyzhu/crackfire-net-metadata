"""Bind a source-only development review; never touch author files."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil

HERE=Path(__file__).resolve().parent
SCI=HERE.parents[2]
sources=[SCI/'latex_v08/main.tex',*sorted((SCI/'latex_v08/sections').glob('*.tex')),
  HERE/'AGENT_v08.md',HERE/'development_prose_review_zh.md',
  SCI/'experiments/research_v08/comparison_plan_v08_420.json',
  SCI/'experiments/research_v08/aggregate_results.py',SCI/'experiments/evaluation/evaluation_stats.py',
  HERE/'engineering_proxy_metrics.py',HERE/'engineering_proxy_metric_checks.json']
snapshot=HERE/'development_prose_snapshot'
snapshot.mkdir(exist_ok=True)
records=[]
for p in sources:
    rel=p.relative_to(SCI);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise ValueError('Snapshot is exclusive-create')
    shutil.copy2(p,dest)
    h=hashlib.sha256(dest.read_bytes()).hexdigest()
    records.append(dict(source_path=str(p),snapshot_path=str(dest),sha256=h,bytes=dest.stat().st_size))
report=dict(review_type='DEVELOPMENT_REVIEW',created_utc=datetime.now(timezone.utc).isoformat(),
  submission_pass=False,source_prose_only=True,pdf_snapshot=None,
  pdf_scope='The manuscript PDF was not yet available for this source-prose review; no page-layout approval inferred.',
  trained_predictions_or_scores_read=False,issues=['D08-P01','D08-P02','D08-P03'],files=records)
(HERE/'development_prose_review_snapshot.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(review_type=report['review_type'],bound_files=len(records),submission_pass=False)))
