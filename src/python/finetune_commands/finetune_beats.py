from pretraining.utils import get_pretrained_weights


resnet34 = get_pretrained_weights(
   checkpoint_file='jobs/beat_multiple_epochs_34_classification/epoch_06_model.weights.h5',
   task='beat',
   arch='resnet34')
resnet34.save_weights('jobs/beat_classification/resnet34.weights.h5')
