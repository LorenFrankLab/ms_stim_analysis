# ms_stim_analysis

This repository accompanies the manuscript **Disruption of theta-timescale spiking impairs learning but spares hippocampal replay (Joshi A, Comrie AE, Bray S, Mankili A, Guidera JA, et al., 2025).**

It provides:

- Ready-to-use scripts to process wideband data into LFP and spikes, and to perform spike sorting, clusterless decoding, and LFP analysis using Spyglass.

- Tools to examine stimulus-driven entrainment and suppression, changes in pairwise correlations, spatial fields, theta sequences, replay, and learning.

- Figure notebooks that reproduce the main and supplementary results.

Results demonstrate the effects of rhythmic and theta-phase-specific closed-loop optogenetic activation of medial septum parvalbumin-expressing neurons on hippocampal LFPs, spatiotemporal coding, and task learning.

**Demo:** Theta-phase-specific stimulation of medial septum PV neurons suppresses the rhythmicity of hippocampal ahead-behind sweeps of location during track traversal.

![Transfected animal](examples/winnie_example_8xslow.gif)

## Installation

To install the package with custom analysis tables and run the associated notebooks (recommended), follow these steps:

1. Clone the repository to your local system.
2. Navigate to the cloned directory and run:
`pip install .`

**Todo**:

- PyPI release

## Usage

### New work

If you want to apply the analysis pipelines to new datasets, you can install the package and use the custom tables together with your existing database and the `spyglass` ecosystem.

### Reuse and Replication

All raw data and derived results (e.g., spike sorting, LFP) will be made available through the DANDI archive *(upcoming)*.

We also plan to release a Docker image that includes:

- a pre-built conda environment
- the notebooks from this repository, and
- a populated SQL database with all information needed to query and retrieve results from the DANDI archive.

(*Docker build in progress*)

### **Associated repositories**

- [non_local_detector](https://github.com/LorenFrankLab/non_local_detector): tools for clusterless decoding of hippocampal population activity.

- [spyglass](https://github.com/LorenFrankLab/spyglass): database framework for managing electrophysiology and behavioral data.

- [trodes](https://bitbucket.org/mkarlsso/trodes/): acquisition and stimulation software used in these experiments.

- [ndx-optogenetics](https://github.com/rly/ndx-optogenetics): NWB extension for representing optogenetic stimulation protocols and metadata.

- [ndx-franklab-novela](https://github.com/LorenFrankLab/ndx-franklab-novela): Frank Lab–specific NWB extension for storing lab-specific data in NWB/DANDI.

### Code Directory

| Figure | Panel | Notebook |
| ------ | ----- | -------- |
| 1 | D | [opto_stimResponse_analysis.ipynb](notebooks/LFP/opto_stimResponse_analysis.ipynb) |
|  | E | [opto_powerSpectrum_analysis.ipynb](notebooks/LFP/opto_powerSpectrum_analysis.ipynb) |
|  | F | [opto_powerSpectrum_analysis.ipynb](notebooks/LFP/opto_powerSpectrum_analysis.ipynb) |
|  | G | [opto_stimResponse_analysis.ipynb](notebooks/LFP/opto_stimResponse_analysis.ipynb) |
|  | H | [opto_powerSpectrum_closedLoop_analysis.ipynb](notebooks/LFP/opto_powerSpectrum_closedLoop_analysis.ipynb) |
|  | I | [opto_powerSpectrum_closedLoop_analysis.ipynb](notebooks/LFP/opto_powerSpectrum_closedLoop_analysis.ipynb) |
| 2 | B | [opto_spiking_response.ipynb](notebooks/Spiking/opto_spiking_response.ipynb) |
|  | C | [mua_response.ipynb](notebooks/Spiking/mua_response.ipynb) |
|  | D | [peak_delay_timescales_lineartrack.ipynb](notebooks/Spiking/peak_delay_timescales_lineartrack.ipynb) |
|  | E | [Fig2E.ipynb](notebooks/Spiking/Fig2E.ipynb) |
|  | F | [place_field_table_plots.ipynb](notebooks/Spiking/place_field_table_plots.ipynb) |
|  | G | [place_field_table_plots.ipynb](notebooks/Spiking/place_field_table_plots.ipynb) |
| 3 | A | [learning_curves.ipynb](notebooks/Behavior/learning_curves.ipynb) |
|  | B | [learning_curves.ipynb](notebooks/Behavior/learning_curves.ipynb) |
|  | C | [learning_curves.ipynb](notebooks/Behavior/learning_curves.ipynb) |
|  | D | [learning_curves.ipynb](notebooks/Behavior/learning_curves.ipynb) |
|  | E | [outbound_error_repeats.ipynb](notebooks/Behavior/outbound_error_repeats.ipynb) |
|  | F | [choice_point_occupancy.ipynb](notebooks/Positions/choice_point_occupancy.ipynb) |
|  | F | [speed_distributions.ipynb](notebooks/Positions/speed_distributions.ipynb) |
| 4 | A | [wtrack_examples_first_epoch.ipynb](notebooks/Decoding/wtrack_examples_first_epoch.ipynb) |
|  | B | [ahead_behind_spectrum.ipynb](notebooks/Decoding/ahead_behind_spectrum.ipynb) |
|  | C | [clusterless_decode_stim_response.ipynb](notebooks/Decoding/clusterless_decode_stim_response.ipynb) |
|  | D | [clusterless_decode_stim_response.ipynb](notebooks/Decoding/clusterless_decode_stim_response.ipynb) |
|  | E | [wtrack_examples_first_epoch.ipynb](notebooks/Decoding/wtrack_examples_first_epoch.ipynb) |
|  | F | [ahead_behind_spectrum.ipynb](notebooks/Decoding/ahead_behind_spectrum.ipynb) |
|  | G | [clusterless_decode_stim_response.ipynb](notebooks/Decoding/clusterless_decode_stim_response.ipynb) |
|  | H | [clusterless_decode_stim_response.ipynb](notebooks/Decoding/clusterless_decode_stim_response.ipynb) |
|  | I | [ahead_behind_spectrum.ipynb](notebooks/Decoding/ahead_behind_spectrum.ipynb) |
| 5 | A | [continuous_traversals.ipynb](notebooks/Decoding/continuous_traversals.ipynb) |
|  | A | [continuous_traversals_first_epoch.ipynb](notebooks/Decoding/continuous_traversals_first_epoch.ipynb) |
|  | B | [opto_ripple_analysis_difference.ipynb](notebooks/Ripples/opto_ripple_analysis_difference.ipynb) |
|  | C | [ripple_decodes.ipynb](notebooks/Ripples/ripple_decodes.ipynb) |
|  | D | [replay_decode_speed.ipynb](notebooks/Ripples/replay_decode_speed.ipynb) |
| S1 | * | [opto_powerSpectrum_analysis.ipynb](notebooks/LFP/opto_powerSpectrum_analysis.ipynb) |
| S2 | A | [supplement_rose_plot.ipynb](notebooks/LFP/supplement_rose_plot.ipynb) |
|  | B | [stim_field_interaction.ipynb](notebooks/Spiking/stim_field_interaction.ipynb) |
| S3 | A | [clusterless_decode_stim_response.ipynb](notebooks/Decoding/clusterless_decode_stim_response.ipynb) |
|  | B | [peak_delay_timescales.ipynb](notebooks/Spiking/peak_delay_timescales.ipynb) |
|  | C | [peak_delay_timescales.ipynb](notebooks/Spiking/peak_delay_timescales.ipynb) |
|  | D | [peak_delay_timescales.ipynb](notebooks/Spiking/peak_delay_timescales.ipynb) |
|  | E | [peak_delay_timescales.ipynb](notebooks/Spiking/peak_delay_timescales.ipynb) |
|  | F | [place_field_table_plots.ipynb](notebooks/Spiking/place_field_table_plots.ipynb) |
|  | G | [decoding_stim_max_ahead_behind_choice_point.ipynb](notebooks/Decoding/decoding_stim_max_ahead_behind_choice_point.ipynb)|
| S4 | * | [ahead_behind_spectrum.ipynb](notebooks/Decoding/ahead_behind_spectrum.ipynb) |

### [Rat alias table](src/ms_stim_analysis/Style/style_guide.py)

| Animal | Alias | Targeted |
| ------ | ----- | -------- |
| Winnie | V | 1 |
| Frodo  | F | 1 |
| Totoro | T | 1 |
| Banner | B | 1 |
| Odins  | O | 1 |
| Wallie | W | 0 |
| Olive  | L | 0 |
| Yoshi  | Y | 0 |
| Bilbo  | I | 0 |
