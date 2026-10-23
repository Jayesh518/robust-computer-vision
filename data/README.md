# Data

Nothing in this folder is committed (see `.gitignore`). Download each dataset from its official source with
`scripts/download_data.py` (or manually, for BDD100K) and keep it under `data/<name>/`. **Do not redistribute any of it.**

## Sources, licenses and decisions (checked 2026-10-23; re-read the primary source before relying on any of this)

| Dataset | Decision | Terms (from the primary source) | Primary source |
|---|---|---|---|
| COCO val2017 | **use** (downloaded) | Annotations: CC BY 4.0. Images: COCO does not own the copyright; use must follow the Flickr Terms of Use and each photo's own license. *Secondary-sourced: COCO's own terms page could not be retrieved.* | https://cocodataset.org/#termsofuse |
| ExDark | **use** (owner's decision, see below) | Terms v1.0 (effective 2026-10-01): free for non-commercial academic research, education, benchmarking and evaluation, including model training. Projects carried out by, for, with or under the sponsorship of a for-profit entity need the licensor's written confirmation first. No redistribution or mirroring. Commercial use, including internal commercial benchmarking, needs written permission. Underlying photos are third-party and not covered. Cite Loh & Chan, CVIU 2019. | `Dataset/DATASET_TERMS.md` in https://github.com/cs-chan/Exclusively-Dark-Image-Dataset |
| DAWN | **use** (non-commercial, no redistribution) | The Mendeley record's structured license is **CC BY-NC 3.0** (attribution, non-commercial). Its description adds: all rights reserved, commercial use and distribution strictly prohibited. Both are satisfied by personal non-commercial research with attribution and no redistribution. Images came from search engines and Flickr, so underlying photo rights remain with their owners. Cite Kenk & Hassaballah, "DAWN: Vehicle Detection in Adverse Weather Nature Dataset", 2020. Sand images are not used. | https://data.mendeley.com/datasets/766ygrbt8y (license read from `data.mendeley.com/public-api/datasets/766ygrbt8y`) |
| BDD100K | **use** (downloaded from a legacy mirror, see below) | UC Regents / BAIR Open Research Commons license, read from the project's own `doc/source/license.rst`: use, copy, modify and distribute for educational, research and not-for-profit purposes, provided the copyright notice and disclaimers accompany every copy. Commercial rights only for BDD/BAIR Commons members. The BSD-3-Clause license in the same repository covers the toolkit code only. | https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst |
| RTTS (RESIDE-beta) | **skip** | The official site states no license or terms (only an informal "feel free to utilize it"). The images were crawled from the web, so third-party rights apply, and the download is a short link plus a Baidu drive with a password. Real haze/fog evidence comes from DAWN (fog) and BDD100K (foggy weather tag) instead. Revisit only if the authors confirm terms in writing. | https://sites.google.com/view/reside-dehaze-datasets |

### Disputed claims, and what I could and could not verify (2026-10-23)
A third-party summary of these licenses disagreed with my checks on three points. Primary sources win:
- **DAWN is CC BY-NC 3.0, not CC BY 4.0.** Read from the Mendeley record itself (`data_licence.short_name = "CC BY NC 3.0"`, link https://creativecommons.org/licenses/by-nc/3.0). The NC clause is why DAWN-trained weights are never released (see the policy below).
- **RTTS: "released strictly for research purposes" could not be confirmed.** Neither the official RESIDE page nor the RESIDE paper's arXiv page (1712.04143) states any license or terms for the dataset. RTTS therefore stays skipped. If I later choose to use it as a local test-only benchmark (standard academic practice), that is a decision to make explicitly, as I did for ExDark.
- **BDD100K is confirmed.** The data and labels are under the UC Regents / BAIR Open Research Commons license (https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst): use, copy, modify and distribute for educational, research and not-for-profit purposes without fee, provided the copyright notice and the two disclaimer paragraphs accompany every copy, modification and distribution; commercial rights only for BDD/BAIR Commons members. The BSD-3-Clause license in that repository covers the toolkit code only. BDD100K has **no animal classes**, so it cannot be the only training set for this project.

### Release policy (what may go on GitHub)
| Item | Policy |
|---|---|
| Code, scripts, configs, tests | publish |
| Metric tables and plots (aggregate numbers) | publish, with dataset citations |
| Dataset files and dataset images | **never** (not in the repository, README, report or portfolio) |
| README / demo visuals | my own photographs or clips, or material whose license explicitly allows this use; faces and number plates blurred |
| Trained weights | publish **only** weights trained on data whose terms I have verified permit it: COCO and BDD100K (include the Regents notice for BDD100K). Never weights trained on ExDark or DAWN. Label releases non-commercial. Ultralytics-derived weights are AGPL-3.0. |

**Consequence for the experiment design:** ExDark and DAWN are **test-only** real sets (evaluation, never training), which also removes any train/test leakage. The A2 variant is trained on COCO plus the adverse-condition splits of BDD100K, and tested on ExDark, DAWN and BDD100K val.

### BDD100K: how the files were obtained, and what could and could not be verified (2026-10-23)
- The documented hosts were unreachable: `bdd-data.berkeley.edu` presents a TLS certificate for the wrong hostname (a fault on their side), and `dl.cv.ethz.ch` does not resolve for anyone (reported upstream in https://github.com/bdd100k/bdd100k/issues/369). Certificate checks were **not** bypassed.
- The legacy Berkeley mirror `http://dl.yf.io/bdd100k/` still serves the files over plain HTTP. Downloaded: `bdd100k_images_100k.zip` (5.5 GB) and `bdd100k_labels.zip` (190 MB), by `scripts/download_data.py bdd100k`.
- **Both files FAIL the md5 published in BDD100K's docs** (labels: got `2d0165fd89b11025206e21a25ae1089d`, documented `e21be3e7d6a07ee439faf61e769667e4`; images: got `c7a1d4db9af5a4691d5f6fee2a62e132`, documented `5a0359c86a0b8713adab1eee9a3041cb`). The mirror's file sizes also differ from the documented sizes, so the docs probably describe a different packaging. **Authenticity is therefore not independently verified.**
- What was checked instead: the image archive has exactly 70,000 train, 10,000 val and 20,000 test images, and every zip member passes its CRC (this detects corruption, not deliberate tampering); the labels hold one JSON per image for the same three splits, each with `weather`, `scene` and `timeofday` attributes; the val attribute counts (5,346 clear, 738 rainy, 769 snowy, 13 foggy) look consistent with the published statistics. Treat BDD results with that caveat.
- **`rider` is mapped to `person`** (`configs/datasets.yaml`), a semantic decision taken before any BDD result was seen: the project is about finding humans, and leaving riders out would score a correct detection of a rider as an error in testing and teach the model to suppress humans in training. `bike` and `motor` are not project classes.
- Only 13 val images are foggy (5 person boxes): BDD gives no usable real-fog evidence.

### Rules this project follows
- **Aggregate numbers only.** Metrics, plots and tables may be published. **Dataset images are not published** (report, repository or portfolio) unless that image's own license allows it. The public failure-case gallery uses my own photographs or clearly licensed images.
- **No redistribution.** Nothing here is mirrored, repackaged or committed (see `.gitignore`).
- **Non-commercial use only.** Models trained or tuned on these datasets are for this personal research and must not be used in any employer or commercial product.
- **Attribution.** Every dataset used is cited in the report (ExDark: Loh & Chan, CVIU 2019; DAWN: Kenk & Hassaballah, 2020; COCO: Lin et al., 2014).

### ExDark: the owner's decision
The ExDark terms require the licensor's written confirmation where a project is carried out "by, for, with or under the sponsorship of a for-profit entity". I am employed by a for-profit company and this project supports my job applications, which is close to that wording. **My decision (2026-10-23): I treat this as personal, non-commercial, educational research (own time, own hardware, no employer involvement, nothing commercial) and use ExDark on that basis, without written confirmation.** The licensor's reading may differ. Safeguards: no redistribution, no publication of ExDark images, no commercial or employer use, and the data and anything derived from it are deleted if the licensor objects. A confirmation email is drafted in the private notes repository (`docs/exdark-permission-email.md`) and can still be sent at any time.

Download: images 1.5 GB and annotations 4.2 MB from Google Drive (ids in `scripts/download_data.py`). Classes: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table. Annotation lines: `<16-char tool prefix> <class> l t w h ...` with pixel coordinates (left, top, width, height). Confirm the class strings against the real files.

## Layout expected by the code
```
data/
  coco/val2017/ + annotations/instances_val2017.json
  exdark/  dawn/  bdd100k/
  processed/<set>/<space>/{images,labels,<space>.yaml}    # generated; <space> is proj or coco80
```
`proj` labels (ids 0..8) are for fine-tuned models; `coco80` labels (COCO indices) are for the COCO-pretrained
baseline. See `configs/classes.yaml` and `src/rcv/labels.py`.

## Known gaps
- No public dataset has real fog/rain/snow images with animal boxes. Those animal results are synthetic only.
- ExDark covers cat and dog in low light only.
- Concrete slabs, rubble and other debris are not COCO classes and no detection dataset for them is in scope for Project 1.
  Occlusion by debris is tested with the synthetic slab in `src/rcv/occlude.py` (a proxy, not real rubble).
- No public dataset is known to this project with thermal images of people under debris, and none with alive/dead labels.
- The exact class strings in each dataset's annotation files must be checked against the downloaded files (`configs/datasets.yaml`).
