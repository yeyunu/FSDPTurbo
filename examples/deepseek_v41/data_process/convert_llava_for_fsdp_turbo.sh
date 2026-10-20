python convert_llava_for_fsdp_turbo.py \
  --input /datasets/llava/llava_instruct_150k.json \
  --output /datasets/llava/processed_llava.json \
  --image-root /data/COCO/COCO2017/train2017 \
  --check-image-exists \
  --drop-missing-image \
  --max-samples 10000
