# ===============> DSADS <===================

python -m src.main \
		with dataset.path="data/dsads" \
		dataset.split_dir="split/dsads" \
		ckpt_path="checkpoints/dsads" \
		dataset.batch_size=256 \
		dataset.jitter=True \
		model.arch='dsads' \
		model.num_classes=19 \
		optim.scheduler="multi_step" \
		epochs=90 \
		trainer.label_smoothing=0.1