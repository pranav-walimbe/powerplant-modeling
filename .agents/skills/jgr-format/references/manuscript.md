# Complete manuscript rules

Use this for research articles in JGR: Machine Learning and Computation. Requirements below apply unless marked **recommended**, **optional**, or **conditional**. See the linked current sources before final submission.

## Journal fit and review identity

JGR-ML seeks original computational/data-driven approaches and scientific results in Earth and space science, including geoscience applications. Explain the novelty and scientific outcome; extensive tutorials on standard ML architectures are unnecessary. This is an editorial expectation, not a requirement to invent a new architecture, add specific experiments, or attain a particular score. [Journal scope][scope]

AGU uses single-anonymized review for this journal: retain author names and affiliations. Confirm originality, no simultaneous consideration elsewhere, and all authors' approval. Preprints are permitted. Disclose relevant conflicts and permissions; AI assistance in writing, images, or data work must identify the tool and use in Methods or a similar section. AI cannot be an author. These confirmations require actual author information. [Policies][policies], [authorship guidance][text]

## Article components and layout

Arrange title/authors/affiliations, key points, abstract, plain language summary, keywords, main text/appendices, acknowledgments, Open Research, and references. Embed figures/tables near their callouts for review rather than collecting them at the end. Use continuous line numbers and 1.5–2 line spacing. [Checklist][checklist], [text requirements][text]

| Component | Rule |
|---|---|
| Title/byline | Brief, informative title; define title abbreviations in the abstract. Give names and numbered institutional affiliations for the work, addresses, and one corresponding author/contact for review. Current-address notes may be separate. |
| Key points | Prepare three complete, self-standing statements; each at most 140 characters including spaces. Avoid abbreviations, acronyms, equations, and special symbols. Detailed guidance permits 1–3; three satisfies the checklist. |
| Abstract | One paragraph, fewer than 250 words. Present question, approach, findings, and contribution; define abbreviations. No figure/table callouts; avoid citations except a justified companion/comment context. |
| Plain language summary | Required for JGR-ML per the checklist; one paragraph, at most 200 words. Explain the problem, work, findings, and relevance without jargon, acronyms, or equations. It must be a distinct explanation rather than an abstract pasted twice. |
| Keywords/index terms | Up to six free keywords and five AGU index terms; index terms are entered at submission. Avoid terms ending in `00`. Profile expertise is a separate field. |

Sources: [summary guidance][summaries], [checklist][checklist], [text requirements][text].

## Text, mathematics, and scientific style

- Use numbered sections and at most four heading levels. Keep heading text descriptive, without citation/figure callouts. Preserve a coherent introduction, methods/data, results, interpretation, and conclusion; exact section names need not be forced into a fixed outline.
- Keep equations editable; define notation and use consistent symbols and cross-references. Define abbreviations independently at first use in the abstract and main text. Footnotes belong only to affiliations and tables. Appendices remain in the article, have titles and introductory text, and use lettered figure/table/equation identifiers such as A1. [Text requirements][text]
- Use SI/metric units throughout; when source measurements use other units, supply verified conversions and retain original-unit context when scientifically useful. Never silently change thresholds or data units. Use consistent chemical subscripts, mathematical typography, abbreviations, date/time conventions, and precision. Date examples use `1 March 2026`, not ambiguous numeric dates. Use a leading zero for fractions. These are style corrections, not permission to change reported measurements. [Style guide][style]

## Figures and tables

- Give figures captions and tables concise titles plus explanatory notes. Number them consecutively and cite them in order; use lowercase panel letters. Maps need latitude/longitude. Keep captions and labels such as `Figure 1` outside graphic artwork. All panels of one numbered figure must form one complete file for production. Recommended formats include PDF/EPS/JPG/TIFF/PS; vector plots should retain legible text. Do not invent a universal DPI or width rule absent current publisher instructions. [Text requirements][text]
- Tables must be editable, with one entry per cell, headings for data columns, and no vertical rules. Use numbered tables, not parts such as 1a/1b. Style concise titles in title case/italics and column headings in sentence case through the template. Put extended explanation in notes; use lowercase superscript letters for specific notes. Image/color-object tables are treated as figures. Prefer notes/bold/italics to color. [Style guide][style], [LaTeX guide][latex]
- **Recommended:** make figures accessible using clear contrast and alternatives to color-only distinctions. Check reduced-size labels, legends, units, scales, and panel letters in the rendered PDF. This review must cover existing assets, not just newly generated ones. [Checklist][checklist]
- Archive the data underlying illustrations and tables, with scripts/workflows needed to reproduce aggregations. Reuse permissions are **conditional** on copyright/licensing; credit sources and document permissions when needed. [Data guidance][data], [policies][policies]

## Citations and bibliography

Use AGU's APA-derived author–date style. Narrative two-author citations use “and”; parenthetical ones use `&`; three or more authors normally use `et al.`. Order grouped citations alphabetically and disambiguate same-author/year items. Let the compatible bibliography style handle details; inspect the result. Protect necessary acronym/proper-name capitalization in BibTeX. [Style guide][style]

References must be publicly accessible before acceptance; ordinary in-preparation/submitted/personal-communication references are unsuitable. Necessary unpublished companions require editor consideration, a copy, and a cover-letter justification. Check text/reference correspondence, DOI/URL accuracy, and supporting-reference inclusion. A preprint is not automatically an unpublished inaccessible manuscript. [Checklist][checklist], [policies][policies]

## Declarations and Open Research

- Acknowledgments identify all relevant funding, including in-kind support and grant identifiers where available. **Optional:** CRediT contributions/additional contribution details. Do not infer roles from author position. Include an explicit conflict statement, including a no-conflicts statement only when confirmed. [Checklist][checklist]
- **Conditional:** address Inclusion in Global Research/local collaborators, permits, authorizations, and Indigenous data governance when applicable. Do not attach these requirements automatically to every satellite/ML study. [Policies][policies], [data guidance][data]
- Put data/software availability in an unnumbered Open Research section before references. Describe objects, repositories, identifiers/URLs, software versions, development links, and access conditions; cite deposited objects in the bibliography. Use version-specific preserved releases, preferably with DOIs. GitHub alone and “available upon request” do not satisfy preservation requirements. Cite third-party data/software too. [Data guidance][data]
- Needed data/code must be accessible during review and preserved for publication. If not public at submission, provide reviewer access/copies through the designated upload type and explain the publication destination. Restricted material needs an editor-approved exception; authors must supply the reason. Supplementary files do not replace a data/software repository. [Data guidance][data]
- In a reproducibility audit, inventory needed processed data, splits, model artifacts/configurations, predictions, and plotting workflows. Describe source-product versions, variables, spatial/time selections, and registration requirements. Determine what supports the reported findings rather than assuming upstream download portals alone preserve this study's research objects.

## Official sources

[scope]: https://agupubs.onlinelibrary.wiley.com/hub/journal/29935210/aims-and-scope
[checklist]: https://www.agu.org/publications/authors/journals/submission-checklists
[text]: https://www.agu.org/publications/authors/journals/text-graphics-requirements
[summaries]: https://www.agu.org/publications/authors/journals/plain-language-summary
[style]: https://www.agu.org/publications/authors/journals/grammar-style-guide
[data]: https://www.agu.org/publications/authors/journals/data-software-for-authors
[policies]: https://www.agu.org/publications/authors/policies
[latex]: https://www.agu.org/publications/authors/-/media/E070CC3AFBE249C680098DCDEBFDCD6A.ashx
