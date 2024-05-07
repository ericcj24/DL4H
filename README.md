# Reproduction of the Paper Transfer Learning for ECG Classification

This file describes the Google Colab implementation to reproduce the results from Transfer Learning for ECG Classification paper.

### Preparation

1. *Project Setup* Data, model weights, and original paper code are stored on Google Drive, the notebook pulls down those information. The directory structure are as follows
    ```python
    # content are organized as:
    # /content/
    #         finetuning/
    #         jobs/
    #         pretraining/
    #         transplant/
    #         beat_classification
    #         dl4h_pack/
    #               transferlearning1.png
    #         data/
    #               physionet/
    #               physionet_test.pkl
    #               physionet_train.pkl
    ```


2. *Project Dependencies* Some key packages' version used in this reproduction. Please note due to CoLab preload a version of cmake that conflicts with original paper's requirement, here we unstall cmake first, then install it as part of a dependency on samplerate package later on.
    ```
    python  3.11
    tensorflow 2.16
    keras 3.3
    ```

    It is recommended to create a virtual environment on local environment

   ```python
       python3 -m venv sixenv
       source sixenv/bin/activate

       pip install --upgrade pip
       pip install -r requirements.txt
   ```



3. *Data* in this project are following three datasets:
    ```
    Icentia11K dataset: original 271.27 GB, in this reproduction, we use 20% of the data
    Physiological Signal Challenge Dataset 2018, in this reproduction, we use all of the data
    PTB-XL database, 1.8GB, in reproduction, we use all 25% of the data
    ```



4. *Run the pretraining job of your choice.* Example is a beat classification job, it will produce output files, such as training history or model checkpoints, that can be found in the `jobs/beat_classification` directory.

    ```shell script
    python -m pretraining.trainer \
        --job-dir "jobs/beat_classification" \
        --task "beat" \
        --train "data/icentia11k_unzipped" \
        --unzipped \
        --arch "resnet18"
    ``` 

5. *Finetuning* Following comand would finetune on af classificaation.
    ```shell script
    python -m finetuning.trainer \
        --job-dir "dl4h_model/mod_jobs/af_classification" \
        --train "data/physionet_train.pkl" \
        --test "data/physionet_test.pkl" \
        --weights-file "dl4h_model/mod_jobs/beat_classification/resnet18.weights.h5" \
        --val-size 0.0625 \
        --arch "resnet18" \
        --batch-size 64 \
        --epochs 200
    ```

6. *Downstream task* On PTBxl dataset
    ```shell script
    python -m finetuning.trainer \
        --job-dir "dl4h_model/mod_jobs/ptbxl_classification" \
        --train "data/ptbxl_train.pkl" \
        --test "data/ptbxl_test.pkl" \
        --weights-file "dl4h_model/mod_jobs/hr_classification/resnet18.weights.h5" \
        --val-size 0.0625 \
        --subset 0.25 \
        --arch "resnet18" \
        --batch-size 32 \
        --epochs 20
    ```

7. *Results* are documented in the jupyter notebook, in reproduction, we showed that pretrained model performaned better than randomly initialized model on downstream task. It also indicates that pretrained model trains faster.
