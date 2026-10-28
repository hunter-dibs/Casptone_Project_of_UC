# Predicting Progression from Mild Cognitive Impairment to Alzheimer's Disease

This capstone project explores whether information collected from people with mild cognitive impairment can help predict who will receive an Alzheimer's disease diagnosis within the following three years. Mild cognitive impairment involves difficulties with memory or thinking.

The project uses existing records from the Alzheimer's Disease Neuroimaging Initiative (ADNI). It compares computer models using different combinations of:

- Memory and thinking test scores, daily activity assessments, and personal background information.
- Measurements from brain scans.
- Protein levels measured in cerebrospinal fluid (CSF).
- Genetic information related to the APOE gene.

## Team Members

- Dibosh Rajbongshi (u3313817) (Project Lead)
- Nirav Navadiya (u3300182)
- Medha Khosla (u3222881)
- Navneet Kaur (u3314485)

## Main findings

The study started with 1,277 participants. Follow-up records supported an outcome for 708 people: 262 received an Alzheimer's diagnosis within three years, while 446 had records supporting no conversion during that period. The remaining participants had unresolved outcomes and were excluded from prediction experiments.

The highest recorded accuracy was **87.93%**, with **153 correct predictions among 174 participants**, using clinical and cognitive information. Each person's prediction was made by a model trained on other participants.

Adding brain scan, CSF, or genetic information helped in some comparisons, but did not always improve predictions. Different experiments used different participant groups because some measurements were missing. No group had complete information from all four sources. The findings need further evaluation using independent data.

## How to read this repository

You do not need to understand programming or install anything to explore the project. Follow this order:

1. **Start with this page.** The overview and main findings explain the research question and what the study found.
2. **See how the study works.** Open the [study process diagram](figures/figure-4-1-methodological-framework.png) to follow the steps from collecting records to comparing predictions.
3. **Understand who was included.** The [participant chart](figures/figure-5-3-cohort-availability.png) shows how many people had enough follow-up information and which measurements were available.
4. **Explore the results.** The [model comparison chart](figures/figure-5-4-model-comparison.png) shows prediction accuracy. The [matched comparison chart](figures/figure-5-5-matched-modality-comparisons.png) shows whether adding another type of information helped when using the same people.
5. **Go deeper if interested.** The [results notebook](src/notebook/notebook.ipynb) contains more charts and tables. The other [notebooks](src/notebook/) explain individual parts of the analysis. GitHub can display saved notebook contents without running the code.

In the charts, **N** means the number of participants, and **accuracy** means the percentage of correct predictions. **C** stands for clinical and cognitive information, **M** for brain scan measurements, **S** for CSF measurements, and **A** for APOE genetic information. A plus sign means those types of information were combined.

Compare results within the same participant group. A higher score in a different group does not, by itself, mean that one model or type of information is better.

## What's in this repository

| Folder | Contents |
|---|---|
| [dataset/](dataset/) | Prepared study data and the participant groups used in the experiments. |
| [src/notebook/](src/notebook/) | Notebooks containing the analysis steps and result summaries. |
| [src/scripts/](src/scripts/) | Programs for preparing data, comparing models, and creating charts. |
| [figures/](figures/) | Charts showing the study process and results. |
| [config/](config/) | Saved settings for the experiments. |

The `requirements` files list the software packages needed to run the analysis.

## About the data

Participant data are kept locally and excluded from the GitHub upload. Running the analysis requires authorized access to the ADNI data and the appropriate software packages. This repository shares the project code, experiment settings, and figures.
