# Demo dataset curation (Day 1 — the linchpin)

You are picking these. This file is the spec the chosen set must satisfy, plus a
reference table you can ignore. The single curation decision drives the PCA hero
visual, the verifier, and the reveal — get it wrong and the clever work is
invisible on the 3-minute video = unscored.

> Unvalidated here: the build sandbox is offline. Confirm every choice fetches or
> falls back cleanly on your machine (`scripts/smoke_fetch.py`) before Day 2.

## The set must do three jobs at once
1. **Clean PCA hero visual** — samples separate into clouds by dataset before
   ComBat, then merge by biology after. This is 30% of the score.
2. **A real metadata mismatch** for the verifier to catch — e.g. survival in
   months vs days, "grade 3" vs "G3" vs "high grade", or a staging-vocab clash.
   Clean data → Claude finds nothing → the 25% verifier loop is invisible.
3. **A batch-masked domain marker** — an HGSOC marker whose signal is masked by
   batch before ComBat and becomes visible after. This is the reveal that crosses
   "cleaned data" → "revealed hidden biology" (the 20% domain edge).

## Hard constraints
- **2–3 datasets, not 5.** Every extra cloud is one more thing that can look wrong
  on camera.
- **Cross-platform, but SAME MODALITY.** Cross-GPL is kept (it's the product's
  pitch), but all demo series must be the same modality — all log-intensity
  microarray. Do NOT mix microarray with RNA-seq counts; ComBat across modalities
  is garbage. (Affy U133A ↔ U133 Plus 2, or Affy ↔ Agilent, are fine.)
- **Shared gene set large enough to plot** after `map_probes` (aim > ~8–10k).
- **ComBat preconditions:** ≥2 batches, enough samples per batch, no covariate
  fully confounded with batch.
- **At least one series that triggers `needs_manual_upload`** is optional but nice
  — proves the fallback is real (honest limitation, not a failure).

## Reference only (a starting point — you're choosing)
Same-modality cross-platform ovarian pairs that historically behave well:
| Accession | Platform (expected) | Note |
|-----------|--------------------|------|
| GSE9891   | GPL570 (Affy U133 Plus 2) | Tothill — large, rich clinical metadata. |
| GSE26712  | GPL96 (Affy U133A)        | Bonome — different Affy chip, same modality; sizeable shared gene set with GPL570. |
| GSE26193  | GPL570 (Affy U133 Plus 2) | Mateescu — has survival; third cloud / metadata contrast. |

## Final chosen set (LOCKED — validated live via series-matrix fetch)

1. **GSE9891** — GPL570, 285 samples, 9 metadata fields (Tothill). Stage uppercase
   (`stagecode`: IIIC/IA), grade numeric, histology as abbreviations
   (Ser/PapSer/Endo/Adeno). Contains LMP (borderline) tumors. No survival fields.
2. **GSE26193** — GPL570, 107 samples, 13 fields (Mateescu). Stage lowercase
   (`stage`: IIIc/Ia), grade numeric, histology full words; OS *and* PFS
   (`os/pfs time (years)` + `os/pfs event` 0/1).
3. **GSE26712** — GPL96, 195 samples, 8 fields (Bonome). No stage field; grade only
   inside free text ("late-stage high-grade"); survival as `survival years` +
   `status` text codes (AWD/NED/DOD). Contains normal-epithelium controls.

Two GPL570 + one GPL96 → cross-platform, same modality (log-intensity microarray).

## Real metadata clashes for the verifier (Day 3 fuel — no seeding needed)
- **Stage:** FIGO casing differs (IIIC vs IIIc); absent in GSE26712.
- **Grade:** numeric field in two; embedded in prose (`tissue`) in GSE26712.
- **Survival:** units (years → target days) + event encoding (AWD/NED/DOD text vs
  0/1) + OS-vs-PFS disambiguation in GSE26193.
- **Histology:** abbreviations (Ser/PapSer) vs full words (Serous).
- **Population:** GSE9891 has LMP borderline; GSE26712 has normal controls — not
  the same comparable cohort.

## Batch-masked marker (Day-5 reveal — decide with merged data in hand)
Choose empirically once merge + ComBat exist (Epic 3): pick an HGSOC gene whose
cross-dataset signal is confounded by batch pre-ComBat and coherent after.
Candidates to check first: WT1, PAX8 (canonical serous markers). Not a Day-2
blocker; validated at Epic 8.4.

## PCA note (Epic 3.4)
Consider restricting the hero PCA to comparable malignant serous tumors (drop
GSE9891's LMP and GSE26712's normals) so the clouds reflect batch, not sample-type
biology. The whole-dataset batch effect will dominate regardless — decide when the
matrix is in hand.
