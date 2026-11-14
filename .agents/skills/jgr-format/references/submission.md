# LaTeX and submission packages

Read [manuscript.md](manuscript.md) for content rules. This reference covers stage-specific files and template conversion; it does not authorize an external submission.

## Template choice and implementation

AGU recommends its templates; initial submission does not require migration merely to use the newest class. When converting, start from the [official 2025 repository][template], preserving the intended review layout. The [older guide][latex] describes 2019 commands and does not override the installed 2025 class.

- Inspect `agujournaltemplate.tex` and the class's selected mode together. The repository contains `draft` and `published` layouts with different definitions; verify which author, correspondence, key-point, and plain-language-summary commands work in the review layout. A sample's commands or comments alone are not proof that metadata renders correctly.
- Replace ACM/CVPR author blocks, title boilerplate, bibliography settings, and unsupported macros as necessary. Load packages supplying needed mathematics/table commands rather than relying on the previous class. Preserve accessible figure descriptions in a compatible form instead of blindly retaining an undefined class-specific command.
- Use author–date citations and the class-compatible bibliography machinery (the supplied class uses `apacite`), not an inherited ACM numbered style. Check narrative versus parenthetical calls. Compile until cross-references and bibliography settle.
- Verify draft spacing/line numbers, including mathematics and captions; do not select `published` solely to mimic a journal PDF. Remove demonstration content and manually forced spacing that makes the review layout unreadable. These are conversion checks, not claims that every class command requires replacement. [Template][template], [class source][class]

## Initial submission

Provide a complete Word/PDF article with figures and tables embedded; compile LaTeX to PDF. Upload supporting information separately. Prepare authors' affiliations/emails, corresponding-author ORCID, at least three suggested reviewers, funding information, and applicable permissions/declarations. Coauthor ORCIDs are encouraged. The cover letter is optional and entered in the submission form, but can explain scope, related manuscripts, and justified exceptions. Never invent reviewers or confirm coauthor approval on their behalf. [Checklist][checklist]

Use the [JGR-ML GEMS portal][portal] and the Research Article category for a full research article. A special collection is optional and must actually fit and accept submissions; journal submission does not depend on selecting one. Verify current portal fields when preparing a live submission.

## Editable-source package

When sources are requested, including at revision:

- Put article text in one main `.tex`; flatten article-text `input/include` chains in the distribution copy. Avoid cross-document `xr` references.
- Use local figure filenames without directory components in the staged article. Supply each numbered figure as one complete file, consolidating all panels. Do not flatten or rename the working repository unnecessarily.
- Include the `.bib` as the appropriate LaTeX bibliography file type, or incorporate the generated `.bbl`. Do not point the bibliography at a subdirectory. Include required nonstandard class/style/support files; standard packages need not be uploaded.
- Mark article, bibliography, and supplementary LaTeX files with the appropriate upload types. Build inside the staged directory; check missing assets, placeholder fallbacks, unresolved references, fonts, and clipped content. [LaTeX guide][latex]
- Tag archived object references with BibTeX `type` values such as `dataset`, `software`, `ComputationalNotebook`, or `Collection`, as appropriate. Complete metadata matters more than choosing a generic web citation. [LaTeX guide][latex]

## Supporting information, if applicable

Use AGU's SI template and submit a final PDF containing supporting text, captions, and illustrations. Upload media separately in accepted native formats. SI is reviewed but not copyedited. It may contain supplementary methods/examples; key analysis and discussion remain in the article. Appendices belong in the main article. [Checklist][checklist], [SI guidance][si]

Data/software belong in trusted repositories, not only SI. Current SI guidance prohibits `.exe`, `.mov`, and macro-bearing `.xlsx` files; verify allowed formats if media or spreadsheets are included. Do not promise arbitrary interactive formats will be accepted. Follow current directions for SI numbering/callouts and ensuring all SI references also appear in the main article. The reference-list inconsistency is explained in `SKILL.md`. [SI guidance][si]

## Revisions and resubmissions

Supply a clean editable manuscript, separate complete figure files, a marked-change Word/PDF, and a point-by-point response reproducing editor/reviewer comments. Include applicable bibliography, SI, availability, and permissions files. Explain any author-list/order changes to the editor and secure actual author agreement. A revision cover letter is optional; the response to reviews is not replaced by it. Recheck current instructions when transferring/resubmitting an already reviewed manuscript. [Checklist][checklist]

## Completion evidence

Report which stage is ready, the exact source/PDF/package checked, the build result, summary limits, PU calculation, and outstanding information. Inspect the PDF as well as logs. Verify that required author metadata and Open Research statements are visible. Do not mark data preservation or permissions complete from placeholder links or unsupported assertions.

## Official sources

[template]: https://github.com/AGU-Publications/agujournal2025-latex-template
[class]: https://raw.githubusercontent.com/AGU-Publications/agujournal2025-latex-template/main/agujournal2025.cls
[latex]: https://www.agu.org/publications/authors/-/media/E070CC3AFBE249C680098DCDEBFDCD6A.ashx
[checklist]: https://www.agu.org/publications/authors/journals/submission-checklists
[si]: https://www.agu.org/publications/authors/journals/supporting-information
[portal]: https://jgr-machinelearning-submit.agu.org/cgi-bin/main.plex
