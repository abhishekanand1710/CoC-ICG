#!/bin/bash
git clone https://github.com/abhishekanand1710/arcade-nl2code.git
cd arcade-nl2code
kaggle datasets download -d googleai/arcade-nl2code-dataset -p arcade_nl2code/annotated_dataset/dataset/

cd arcade_nl2code/annotated_dataset/dataset
unzip arcade-nl2code-dataset.zip

pip install -r arcade_nl2code/evaluation/requirements.txt
pip install seqio nbconvert diff-match-patch dacite

cd ..
PYTHONPATH=../../ python build_existing_tasks_split.py

# cd arcade-nl2code/arcade_nl2code/annotated_dataset
PYTHONPATH=../../ python build_new_tasks_split.py