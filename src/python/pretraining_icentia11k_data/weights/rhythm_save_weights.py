from pretraining.utils import get_pretrained_weights
resnet18 = get_pretrained_weights(
   checkpoint_file='jobs/rhythm_classification/epoch_01_model.weights.h5',
   task='rhythm',
   arch='resnet18')
resnet18.save_weights('jobs/rhythm_classification/resnet18.weights.h5')
