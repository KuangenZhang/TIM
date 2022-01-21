
# =============> Resnet18 <================
python -m src.main \
		-F logs/tim_gd/cub/ \
		with dataset.path="data/dsads" \
		ckpt_path="checkpoints/dsads" \
		dataset.split_dir="split/dsads" \
		model.arch='dsads' \
		model.num_classes=19 \
		tim.iter=1000 \
		evaluate=True \
		eval.method='tim_gd' \

