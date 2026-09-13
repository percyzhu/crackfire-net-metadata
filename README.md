# CrackFireNet: fire-exposed timber beam benchmark

Data and reproducibility materials accompanying **Local geometry-aware crack-graph learning for section-modulus prediction in fire-exposed timber beams**, by Xiuzhi Zheng, Bo Peng and Shaojun Zhu, Tongji University.

Given known crack geometry, member dimensions and a prescribed fire history, the model predicts four directional remaining geometric section-modulus histories. Section-level outputs support selection of weak regions and comparison with a prescribed geometric demand.

## Download the complete study

**[Open the current manuscript and complete data release](https://github.com/percyzhu/crackfire-net-metadata/releases/tag/eaai-benchmark)**

| Material | Direct download |
|---|---|
| Complete learning data, models, selected weights, saved predictions and reproduction scripts | [EAAI_reproducibility.zip](https://github.com/percyzhu/crackfire-net-metadata/releases/download/eaai-benchmark/EAAI_reproducibility.zip) |
| Main manuscript | [Manuscript.pdf](https://github.com/percyzhu/crackfire-net-metadata/releases/download/eaai-benchmark/Manuscript.pdf) |
| Supplementary material | [Supplementary_material.pdf](https://github.com/percyzhu/crackfire-net-metadata/releases/download/eaai-benchmark/Supplementary_material.pdf) |
| Editable LaTeX manuscript, figures and plotting source | [EAAI_manuscript_LaTeX.zip](https://github.com/percyzhu/crackfire-net-metadata/releases/download/eaai-benchmark/EAAI_manuscript_LaTeX.zip) |
| Submission highlights | [Highlights.docx](https://github.com/percyzhu/crackfire-net-metadata/releases/download/eaai-benchmark/Highlights.docx) |
| Asset SHA-256 checksums | [SHA256SUMS.txt](https://github.com/percyzhu/crackfire-net-metadata/releases/download/eaai-benchmark/SHA256SUMS.txt) |

The complete learning package is distributed as a GitHub Release asset. **GitHub's automatically generated “Source code” archives contain only this repository checkout**, not the complete release dataset.

## Benchmark at a glance

- **1,482 individually simulated members:** 982 reference-size and 500 variable-size cases.
- Inputs: actual member dimensions, individual crack geometry, directed pair relations, masks and complete 61-point fire curves.
- Targets: `[1482, 361, 31, 4]` section-modulus ratios on 0–3600 s; member curves are the directional minimum over the 31 standard sections.
- All **1,482 compact first-300°C crossing records** and material masks allow local reconstruction of the published remaining-section labels.
- Five model variants, three seeds, **15 selected checkpoints**, **90 standard-grid prediction files**, and **420 dense-coordinate diagnostic prediction files**.
- Final geometry-family partition, all source identifiers, preprocessing and generation scripts, material tables, evaluation metrics and frozen resampling indices.

The final partition contains 860 training, 27 validation, 208 count-transfer, 80 section-transfer, 70 length-transfer, 20 joint-transfer, 10 reserved and 207 isolated-relative cases. The earlier 12 partitions are retained as historical definitions inside the complete package; they are not the final experiment partition.

## Reproduce

Download and extract `EAAI_reproducibility.zip`, enter the extracted directory and run:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The command validates file hashes, checks all selected checkpoints, reconstructs sample geometric labels, re-scores every published prediction, recalculates paired family intervals and recreates the main results table and core figures. Outputs go to `reproduced/`; this command does not launch thermal simulation or training. Python 3.12 and PyTorch CPU are supported. Figure text uses Arial, which must be available locally and is not redistributed here; mathematics is typeset separately.

Read [dataset_card.md](dataset_card.md), [reproducibility/README.md](reproducibility/README.md) and [schema.json](reproducibility/schema.json) before analysis. The small `reproducibility/` folder exposes the portable entry points for inspection; execute them from the downloaded complete package.

## Target definition and source fields

The target is the remaining **geometric** section-modulus ratio after irreversible first reaching 300°C, calculated using each member's actual dimensions and the stated numerical model. It is not an experimentally measured resistance or a mechanically reduced effective-section capacity. The section grid is part of the target definition: axial sampling affects minima and first-crossing reference times.

Full original meshes and nodal thermal fields remain in the indexed research archive. The downloadable package supplies every compact record needed to reconstruct the published labels, plus mesh/field hashes and source readers. Dense-coordinate diagnostics and original numerical audit records retain their stated scope and status.

## Earlier study materials

The pre-existing `reproducibility-v08/` directory and root-level historical manifests are preserved for traceability. They describe a different, earlier target and experiment. See [HISTORICAL_STUDY.md](HISTORICAL_STUDY.md). Use the release linked above for the current manuscript; do not merge the historical experiment tables into the present benchmark.

## Attribution and contact

Please cite the associated manuscript and this repository using [CITATION.cff](CITATION.cff). The existing [CC BY 4.0 license](LICENSE) and any per-file third-party license/attribution continue to apply. Font files and third-party dependencies are not bundled. See the package's third-party notices.

Supported by the National Natural Science Foundation of China (Grant No. 52408206).

Shaojun Zhu, corresponding author, Tongji University — zhushaojun@tongji.edu.cn.
