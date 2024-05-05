python -m finetuning.trainer \
--job-dir "jobs/beat_random_classification" \
--train "data/physionet_train.pkl" \
--test "data/physionet_test.pkl" \
--val-size 0.0625 \
--arch "resnet18" \
--batch-size 64 \
--epochs 200
