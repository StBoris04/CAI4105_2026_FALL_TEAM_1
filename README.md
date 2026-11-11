# CAI4105 Team 1

This repository holds our work for two machine learning challenges. Each
challenge has its own folder so data, code, validation results, and submissions
stay separate.

## Repository layout

| Folder | Challenge | Target |
| --- | --- | --- |
| [`challenge1/`](challenge1/README.md) | Apartment rent prediction | `price` |
| [`challenge2/`](challenge2/README.md) | Online shopping purchase prediction | `Revenue` |

Challenge-specific setup and run instructions are in each folder's README.

## Challenge 1

The rent prediction workflow, training and test data, notebook, archive, and
existing `ID,price` submission are in `challenge1/`. Run the script from that
folder with `python "FIRST DRAFT MODEL.py"`.

## Challenge 2

The Challenge 2 workspace is prepared under `challenge2/`. When the instructor
CSV files are available, place `challenge2_train.csv` and `challenge2_test.csv`
in `challenge2/data/`. These course data files are excluded from Git. The
modeling workflow will use `Revenue=True` F1 as its primary validation metric,
compare the required classifiers, tune selected models with Grid Search, and
write the final `ID,Revenue` submission to `challenge2/submissions/`.

See [`challenge2/README.md`](challenge2/README.md) for the full modeling plan
and setup details.
