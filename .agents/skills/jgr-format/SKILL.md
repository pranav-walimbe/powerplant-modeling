---
name: jgr-format
description: Audit, format, or prepare research articles for AGU Journal of Geophysical Research - Machine Learning and Computation, including manuscript structure, LaTeX conversion, figures, references, Open Research, submission files, and publication-unit counting. Use for JGR-ML research-article preparation, not AGU meeting abstracts or Geophysical Research Letters.
---

# JGR research-article formatting

Prepare research articles for **Journal of Geophysical Research: Machine Learning and Computation** (JGR-ML). Apply the complete applicable rules, not a checklist limited to previously identified defects.

## Scope and authority

- Identify whether the request is an audit, manuscript edit, initial submission package, revision, or length calculation. Audit requests do not authorize manuscript edits; a request to format or prepare the paper does authorize the relevant local changes. Creating this skill does not authorize editing the paper.
- Keep scientific meaning, metrics, labels, results, author order, and attribution intact. Flag substantive issues separately from formatting. Do not invent grants, contribution roles, conflicts declarations, author approvals, repository deposits, DOIs, or reviewer identities.
- Local preparation does not itself authorize submitting, depositing/publishing research objects, paying fees, or contacting editors/coauthors.
- Journal-specific current instructions and the live submission form take precedence over general AGU guidance; distinguish actual requirements from recommendations and conditional policies. Verify changing requirements before declaring a package ready. Snapshot reviewed: **14 November 2026**.
- The [journal author instructions](https://jgr-machinelearning-submit.agu.org/cgi-bin/main.plex?form_type=display_auth_instructions) link to the authoritative AGU resources. Use official AGU/Wiley pages and the official template repository for updates.

## Read the relevant reference

- For every full manuscript audit or formatting task, read [manuscript.md](references/manuscript.md). This covers the entire article, declarations, citations, illustrations, and Open Research.
- For template conversion, submission packaging, supporting information, or revisions, also read [submission.md](references/submission.md).
- For word counts, length advice, or a complete readiness check, read [publication-units.md](references/publication-units.md). Recalculate from the current manuscript; never reuse a previous count.

## Resolve known documentation conflicts

1. **Plain language summary:** the current submission checklist explicitly requires one for JGR-ML, although another general AGU page omits this journal. Include it unless newer journal-specific instructions say otherwise.
2. **Template versions:** the linked seven-page LaTeX guide describes `agujournal2019`; the official repository supplies `agujournal2025`. Use guidance for the chosen class and build mode, inspecting the actual class when necessary. Do not mix incompatible author, key-point, or summary commands from different versions or modes.
3. **Length:** 25 publication units is a general AGU reference point, not a hard JGR-ML rejection threshold. JGR-ML is fully open access with no excess-length fee. The 12-PU research-letter limit is not this article's limit.
4. **Supporting-information references:** the checklist requests a main-article section called `References From the Supporting Information`, while other guidance favors a combined reference list and varies on explicit main-text callouts. Keep every supporting reference discoverable in the main article, follow the current checklist/editor directions, and flag any remaining conflict instead of silently dropping references.

## Working procedure

1. Inspect the actual article source, bibliography, figure assets, existing build commands, repository instructions, and working-tree changes. In this repository, start by checking `assets/paper/paper.tex`; do not substitute the CVPR version or preprint PDF without checking the user's intended version.
2. For an audit, report each applicable requirement as satisfied, missing, needing verification, or not applicable. Keep optional recommendations separate. Include the source and stage for disputed or conditional rules.
3. When edits are requested, make the necessary changes and build the review PDF. Put distribution-only file/path changes in a staging package when appropriate. Keep code, datasets, and scientific content outside the requested formatting scope intact.
4. Review the rendered PDF: front matter, page/line numbers, spacing, equations, citations, figure/table placement and legibility, and final declarations. Inspect build logs for unresolved citations/references, missing assets, font failures, and overflows that affect readability. `IfFileExists` fallbacks must not hide missing final figures.
5. Recount summaries and publication units after changes. Validate the submission package from its own directory so local paths cannot mask missing dependencies. Use meaningful build and visual checks rather than prose-matching tests.
6. Report completed changes, validation, current word/figure/table counts, and any remaining author-supplied information or external actions. A successful compile alone does not establish journal compliance.
