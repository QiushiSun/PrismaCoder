export MODELSCOPE_CACHE=/path/to/ms-cache
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export IMAGE_MAX_TOKEN_NUM=2048
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=16
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=14
export NNODES=4

OUTPUT_BASE=/path/to/output/dir

megatron sft \
    --model /path/to/Qwen3.5-35B-A3B \
    --save_safetensors true \
    --dataset /path/to/dataset/folder \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --tuner_type full \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --context_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-6 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 5 \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --output_dir ${OUTPUT_BASE} \
    --save_steps 2000 \
    --max_length 32768 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 32 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --packing true \
    --sequence_parallel false
