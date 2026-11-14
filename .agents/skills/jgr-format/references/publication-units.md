# Publication-unit calculation

Calculate from the current article, not the PDF page count, source-code word count, image-file count, or an earlier conversation's estimate. [AGU's fee/length page][fees] controls the count categories.

`PU = counted words / 500 + numbered figures + tables`

| Include in counted words | Exclude |
|---|---|
| Abstract; main text and headings; in-text citations; figure/table captions; acknowledgments; appendices; equations at one word each | Title; author names/affiliations; keywords/key points; plain language summary; table-cell text; Open Research; bibliography; supporting information |

Count a multipart numbered figure once, even if its working source uses multiple graphics. Appendix figures/tables are part of the main paper; SI figures/tables are not. Keep fractional PU precision and distinguish the exact result of the chosen method from the publisher's final calculation.

## Reproducible counting method

Use a temporary counting copy or explicit parser exclusions. Do not change the submitted manuscript merely to measure it. AGU's [LaTeX guide][latex] suggests TeXcount after removing excluded sections.

1. Record the article/version and exclusions. Identify numbered figure/table environments, including starred environments and appendices, without counting commented examples.
2. Use TeXcount or rendered-text counting with the inclusion rules above. Keep captions while excluding table cells and figure artwork/alternative-description metadata. Do not count placeholder branches or both sides of conditional markup.
3. Configure custom macros and citations explicitly. TeXcount defaults may ignore emitted words from custom commands and citations. Count actual rendered author–date citations when possible; if treating each current numeric citation group as one token, state that convention and expect the count to change on conversion.
4. Count each equation as one word. A multirow `align` can contain several equations, despite being one environment. Normalize attached chemical subscripts such as `NO$_2$` as part of the chemical word rather than an extra equation. Exclude math inside excluded table cells. State borderline token conventions instead of presenting spurious precision.
5. Separate text, headings, captions, equation contributions, figure count, and table count in the report. Give the final formula, a suitably rounded PU, counting limitations, and the summary counts. Recalculate after formatting/content changes.

For TeXcount, `-sum=1,1,1,0,0,1,1` includes text, headings, captions, inline mathematics, and display mathematics, with no float/header-count surcharge. It still requires the exclusions, macro/citation handling, and equation adjustments above. Inspect subcounts before accepting the total; raw `-sum` is not automatically AGU compliant.

## Length interpretation

Twenty-five PU is AGU's general research-article reference point. Longer JGR-ML research articles are permitted; do not apply the 12-PU research-letter cap. JGR-ML is fully open access, with no base publication or excess-length fee; an APC may apply on acceptance. Verify current fees and funding agreements rather than storing a price or institutional discount as a permanent rule. Do not move key interpretation into SI merely to reach a round number. [Fee/length policy][fees], [article types][types]

## Official sources

[fees]: https://www.agu.org/publications/authors/journals/publication-fees
[latex]: https://www.agu.org/publications/authors/-/media/E070CC3AFBE249C680098DCDEBFDCD6A.ashx
[types]: https://www.agu.org/publications/authors/journals
