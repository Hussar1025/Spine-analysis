# Spine-analysis: Intervertebral Disc Degeneration and LIPUS Intervention Prediction Model

## 1. Background
Intervertebral disc degeneration (IVDD) is a crucial pathological foundation for degenerative cervical spine diseases. Current treatments primarily focus on symptom management and lack effective interventions to explicitly delay the progression of degeneration. Low-Intensity Pulsed Ultrasound (LIPUS) offers advantages such as non-invasiveness, repeatability, and adjustable parameters. However, cellular responses to LIPUS vary significantly, and there is a critical lack of a pre-intervention prediction system targeting the ultrasound sensitivity of degenerated nucleus pulposus (NP) cells.

## 2. Design Concept
This project shifts the paradigm from "post-intervention validation" to "pre-intervention prediction." By integrating AFM static mechanical parameters, micro-nano optical dynamic vibration features, single-cell omics, and pseudotime trajectory information, we aim to construct a prediction model for the LIPUS intervention response of degenerated NP cells.

## 3. Experimental Content
* **Clinical Sample Processing:** Collection of clinical degenerated NP samples, establishing standardized cell isolation, culture, and stratification protocols.
* **Mechanical Profiling:** Utilization of AFM to detect static mechanical indicators (e.g., elastic modulus), combined with micro-nano optical platforms to acquire dynamic mechanical characteristics (e.g., vibration spectra, frequency response), supported by Finite Element Analysis (FEA).
* **Single-cell Analysis:** Execution of single-cell sequencing, cell subpopulation identification, differential analysis, pathway enrichment, and pseudotime trajectory reconstruction to extract key molecular features related to degeneration and potential reversibility.
* **In Vitro Intervention:** Implementation of LIPUS intervention in primary NP cells and definition of response labels.
* **Model Construction:** Establishment of a multi-modal fusion prediction model based on the above data.
* **In Vivo Validation:** Completion of validation using mouse IVDD models.

## 4. Dry Lab Design
1. **Epidemiological Analysis:** Using the GHDx database (1990-2022) for China's cervical spine disease data, employing ARIMA and other methods to analyze and predict prevalence trends.
2. **Pseudotime Modeling:** * **Trajectory Reconstruction:** Applying `monocle3` to public spinal NP cell datasets for pseudotime analysis.
    * **OT-Supervised Alignment:** Constructing cell niches and utilizing Optimal Transport (OT) to align $t_0$ and $t_1$ time-series omics data. This alignment acts as a supervised constraint to interpolate dynamic changes between these two time points and effectively align pseudotime with real-world physical time.
    * **GNN Prediction:** Constructing an unsupervised cell fate trajectory prediction model using GNNs, further optimized by the supervised real-time data to build a high-precision NP cell fate model for tracking key gene dynamics.
3. **Finite Element Analysis:** Simulating the mechanical response of cultured NP cells to ultrasonic vibration at different time points to verify mechanical differences and track response trends.
4. **Systems Biology Modeling:** Manually integrating known LIPUS-related pathways to predict dynamic changes in key genes and signaling molecules during intervention.
5. **Multi-modal Integration:** Aligning pseudotime omics predictions with mechanical phenotype predictions. Cross-validating system biology simulation results with omics-tracked key gene changes to integrate "mechanical phenotype-omics" multi-modal data, constructing the final NP cell degeneration and LIPUS response prediction model.

## 5. Repository Structure
* `code/`
    * `Pseudotime/` - Scripts for monocle3, GNN, and OT-alignment.
    * `Epidemiology/` - ARIMA modeling scripts for GHDx data.
* `result/`
    * `Pseudotime/` - Model outputs and analysis results.
    * `Epidemiology/` - Epidemiological forecast results.
* `data/`
    * `Pseudotime/` - Public database and our own database

## 6. Data Description
We utilize multi-source datasets to support our predictive modeling, including single-cell omics data, cellular AFM mechanical data, micro-nano optical data, and public database resources (e.g., GBD, NIH).
* **Core Datasets:**
    * GSE244889 (including sample GSM7831813 and others)
    * Xenium_V1_hHeart_nondiseased_section_FFPE
