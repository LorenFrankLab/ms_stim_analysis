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
