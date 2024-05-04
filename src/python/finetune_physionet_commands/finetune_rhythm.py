python -m finetuning.trainer \
--job-dir "jobs/rhythm_classification" \
--train "data/physionet_train.pkl" \
--test "data/physionet_test.pkl" \
--weights-file "jobs/rhythm_classification/resnet18.weights.h5" \
--val-size 0.0625 \
--arch "resnet18" \
--batch-size 64 \
--epochs 200
