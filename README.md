# BioLaySumm2025

We extrat 100 samples from the training data as the test samples. As shown in `./BioLaySumm2025-eLife_result.json`

## Installing



For the evaluation, we need to create 3 different environments to run script of different metrics.



### general_metric

This environment is used for runing `./evaluation.py` and `./evaluation_AlignScore.py`



Please create environment by folloing code:

```
conda create -n general_metric python=3.9.0

conda activate general_metric 

pip install nltk

pip install rouge

pip install scipy

pip install gritlm

pip install textstat

pip install transformers
```



Run evaluation:

```
python evaluation.py --prediction_file  BioLaySumm2025-eLife_result.json  --groundtruth_file BioLaySumm2025-eLife_result.json --task_name Lay_Summarisation  

python evaluation_AlignScore.py --prediction_file  BioLaySumm2025-eLife_result.json  --groundtruth_file BioLaySumm2025-eLife_result.json --task_name Lay_Summarisation  
```



### SummaC

This environment is for SummaC

```
conda create -n SummaC python=3.9.0

conda activate SummaC

pip install torch

pip install summac
```

Evaluation:

```
python evaluation_SummaC.py --prediction_file  BioLaySumm2025-eLife_result.json  --groundtruth_file BioLaySumm2025-eLife_result.json --task_name Lay_Summarisation  
```

### RagGraph

It is used for **F1RadGraph** and **F1ChexBert**

```
conda create -n RagGraph python=3.9.0

conda activate RagGraph 

pip install torch==2.3
pip install transformers==4.39.0
pip install appdirs
jpip install sonpickle
pip install filelock
pip install h5py
pip install spacy
pip install nltk
pip install pytest

pip install scikit-learn
pip install numpy
pip install appdirs
pip install pandas

pip install radgraph

pip install f1chexbert
```

Evaluation:

```
python evaluation_f1radgraph.py --prediction_file  BioLaySumm2025-eLife_result.json  --groundtruth_file BioLaySumm2025-eLife_result.json --task_name Radiology_Report_Generation  

python evaluation_f1chexbert.py --prediction_file  BioLaySumm2025-eLife_result.json  --groundtruth_file BioLaySumm2025-eLife_result.json --task_name Radiology_Report_Generation
```

