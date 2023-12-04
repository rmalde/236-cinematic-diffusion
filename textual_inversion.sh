export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="images/train_epic_no_metadata"
export OUTPUT_DIR="outputs/models_epic_tinv"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<gangnam-style>" \
  --initializer_token="epic" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=1e-3 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR