python -m finetuning.trainer \
--job-dir "jobs/ptbxl_random_classification" \
--train "data/ptbxl_train.pkl" \
--test "data/ptbxl_test.pkl" \
--val-size 0.0625 \
--subset 0.25 \
--arch "resnet18" \
--batch-size 32 \
--epochs 20