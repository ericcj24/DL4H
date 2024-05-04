from pretraining.utils import get_pretrained_weights
resnet18 = get_pretrained_weights(
   checkpoint_file='jobs/hr_classification/epoch_01_model.weights.h5',
   task='hr',
   arch='resnet18')
resnet18.save_weights('jobs/hr_classification/resnet18.weights.h5')
