# Competent Programmer Assumption for Heuristic Patch Correctness Assessment in Automated Program Repair

This repository contains our datasets of patches, measurements, and analyses.
Directory structure of this repository is as follows.
```plain
/
|
+--- Competent Programmer Assumption
    |
    +-- Patches
    |
    +-- Measurements
    |
    +-- Analyses (RQ 1-2)
+--- Mutation-based Patch Correctness Assessment
    |
    +-- Measurements
    |
    +-- Patches
    |
    +-- RQ1
    |
    +-- RQ2
    |
    +-- QuixBugs Projects
```

The directory `Competent Programmer Assumption` contains all the files for the first half of the paper (i.e., empirical study of competent programmer assumption),
while `Mutation-based Patch Correctness Assessment` contains the files related to the second half of the paper (i.e., our new mutation-based patch correctness assessment techique).

## Competent Programmer Assumption
The sub-directory `Patches` contains all the Defects4J patches studied in this part.
The sub-directory `Measurements` contains raw Jaccard similarity and cosine similarity measurements for each patch.
Last but not least, the sub-directory `Analyses (RQ 1-2)` contains a `.ods` contains all our statistical analyses.

## Mutation-based Patch Correctness Assessment
The sub-directory `Measurements` contains raw Jaccard similarity and cosine similarity measurements for each patch.
It also contains measurements for ODS and Shibboleth for cross-validation.
The sub-directory `Patches` contains all the QuixBugs patches studied in this part.
The sub-directory `RQ1` contains all the Python scrips for various learners studied in RQ1 in Section 4.2 of the paper.
The sub-directory `RQ2` contains all the Python scrips for various learners studied in RQ2 in Section 4.2 of the paper.
Last but not least, the sub-directory `QuixBugs Projects` contains all the Java QuixBugs bugs.

## System Requirements
To run Python script, we used a Macbook Pro with Python 3.9.
Our scripts expect the following libraries: (1) scikit-learn, (2) numpy, (3) pandas, and (4) xgboost.
