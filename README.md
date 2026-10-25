<h1 align="center">Detecting Changes in Power Plant NO<sub>x</sub> Emissions Using Deep Learning and TEMPO Satellite Imagery</h1>

<p align="center">
  <a href="https://eartharxiv.org/repository/view/15747/">
    <img src="https://img.shields.io/badge/EarthArXiv-Preprint-247BA0" alt="Read the preprint on EarthArXiv">
  </a>
</p>

## What it does

- Classifies short-term power plant emissions trajectories as **increasing**, **steady**, or **decreasing** from five consecutive TEMPO observations.
- Learns relative emissions behavior, with labels derived from aggregate EPA CEMS emissions within each plant-centered area of interest (AOI).

## Inputs

- **Satellite observations:** five TEMPO NO2 rasters, each 24 × 24 pixels over a 72 × 72 km AOI, with pixel-validity information.
- **Meteorology:** HRRR surface temperature and wind components.
- **Transport residuals:** differences between observed NO2 and the preceding plume's predicted transport and decay.
- **Time features:** sine/cosine of local solar hour and day of year.

<p align="center">
  <a href="assets/paper/visuals/sample-construction.pdf">
    <img src="assets/readme/sample-construction.png" width="1000" alt="Example sequence with NO2 observations, meteorology, transport residuals, and time features.">
  </a>
</p>

## Model architecture

- Separate convolutional encoders process observation rasters and transport residuals.
- A **ConvGRU** combines spatial features across the sequence; the final representation is fused with time features to predict the three emissions-change classes.

<p align="center">
  <a href="assets/paper/visuals/convgru-architecture.pdf">
    <img src="assets/readme/convgru-architecture.png" width="1000" alt="ConvGRU with observation and transport-residual encoders, recurrent updates, pooling, and late fusion with time features.">
  </a>
</p>

## Results

- On **22,466 held-out sequences** covering **116 active U.S. power plants**, ConvGRU achieves **0.56 test macro-F1**, compared with **0.45** for a time-only MLP baseline.
- ConvGRU improves F1 across all three classes; transport residuals contribute the most in the input ablation study.

<p align="center">
  <a href="assets/paper/visuals/classification-performance-test.pdf">
    <img src="assets/readme/classification-performance-test.png" width="1000" alt="Test F1, recall, and macro-F1 for the baseline MLP and ConvGRU.">
  </a>
</p>

[Setup and pipeline instructions](assets/docs/setup-and-pipeline.md) · [Paper PDF](assets/paper/paper.pdf) · [CC BY 4.0 license](LICENSE)

## Citation

```bibtex
@misc{walimbe2026tempochange,
  title = {Detecting Changes in Power Plant NOx Emissions Using Deep Learning and TEMPO Satellite Imagery},
  author = {Walimbe, Pranav and Murayama, Hikari and Cohen, Ronald C.},
  year = {2026},
  publisher = {EarthArXiv},
  doi = {10.31223/X50J7K},
  url = {https://eartharxiv.org/repository/view/15747/}
}
```
