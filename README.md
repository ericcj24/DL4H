# Reproduction of the Paper Transfer Learning for ECG Classification

This file describes the Google Colab implementation to reproduce the results from Transfer Learning for ECG Classification paper.

### Preparation

1. *Data, model weights, and original paper code are stored on Google Drive, the notebook pulls down those information

2. *Some key packages' version used in this reproduction.
    ```python
    tensorflow 2.16
    keras 3.3
    ```

3. *Run the pretraining job of your choice.* Let's run a beat classification job which will produce output files, such as training history or model checkpoints, that can be found in the `jobs/beat_classification` directory. Notice the `--arch` option that we used to specify ResNet-18 as the architecture that we want to pretrain. For more options, see `pretraining/trainer.py`.

    If you decided not to unzip the files, but rather want to unzip them on the fly during training, then remove the `--unzipped` option and change the `--train` option to `data/icentia11k`. 

    ```shell script
    python -m pretraining.trainer \
    --job-dir "jobs/beat_classification" \
    --task "beat" \
    --train "data/icentia11k_unzipped" \
    --unzipped \
    --arch "resnet18"
    ``` 

4. *Save weights for finetuning.* Now that we have pretrained our ResNet-18, let's extract its weights from a model checkpoint into a separate file which we will later use for finetuning. However, first we need to choose the model checkpoint that we want to get the weights from. A solid choice is a checkpoint that scored good on our validation metric. 

    Since we have trained our network for only one epoch in the step above, which produced only one checkpoint `jobs/beat_classification/epoch_01/model.weights`, we will simply use that checkpoint. After running the code snippet below, our weights will be stored in the `jobs/beat_classification/resnet18.weights` file.

    ```python
   from pretraining.utils import get_pretrained_weights
   resnet18 = get_pretrained_weights(
       checkpoint_file='jobs/beat_classification/epoch_01/model.weights',
       task='beat',
       arch='resnet18')
   resnet18.save_weights('jobs/beat_classification/resnet18.weights')
    ```

5. That's it! See [finetuning](../finetuning) for instructions on how to finetune our pretrained network to a different task.
