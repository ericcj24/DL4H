from pretraining.utils import get_pretrained_weights
resnet18 = get_pretrained_weights(
   checkpoint_file='jobs/beat_classification/epoch_01_model.weights.h5',
   task='beat',
   arch='resnet18')
resnet18.save_weights('jobs/beat_classification/resnet18.weights.h5')
