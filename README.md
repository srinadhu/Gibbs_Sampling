Gibbs_Sampling
======================

Introduction
------------

In this project density estimation using Gibbs Sampling is done. We are provided with [Adult income dataset](https://archive.ics.uci.edu/ml/datasets/adult) as train data and test data. We are already provided with BayesNet on the train data. Using this Bayes Net, Gibbs Sampler will generate samples, then for each data-point in test data probability with Bayes Net and probability from sample generation will be compared. Mean squared error is used as error measure.

Directory Structure
-------------------

---Sampler

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Gibbs_Sampling.py](Sampler/Gibbs_Sampling.py)

---[lab.pdf](lab.pdf)

---README.md

---[report.pdf](report.pdf)

Executing
---------

After fixing the number of samples and burn\_in, run $python Gibbs_Sampling.py

Developed by
------------
[Sai Srinadhu K](https://www.linkedin.com/in/sai-srinadhu-katta-a189ab11b/)
