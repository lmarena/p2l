import argparse
import os
import yaml
import json
import random
from transformers import Trainer, TrainingArguments, set_seed
from p2l.dataset import DataCollator, get_model_list, get_dataset, translate_val_data
from p2l.model import get_p2l_model, get_tokenizer
from torch.utils.data import Sampler
from typing import Optional
from huggingface_hub import HfApi

# Want control over data ordering, use no shuffle trainer.
class NoShuffleTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[Sampler]:
        return None


def train_model(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    learning_rate = config["learning_rate"]
    # Microbatch size
    batch_size = config["batch_size"]
    # HF data path
    train_data_path = config["train_data_path"]
    val_data_path = config["val_data_path"]
    output_dir = config["output_dir"]
    pretrain_model_name = config["pretrain_model_name"]
    # Prompts will be truncted to this length
    max_length = config["max_length"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    # Deepspeed config choices can be found in the deepspeed directory
    deepspeed_config_path = config["deepspeed_config_path"]
    # Type of transformer, see model.py for options.
    model_type = config["model_type"]
    # Loss type (e.g, bt, rk), see model.py for options.
    loss_type = config["loss_type"]
    # The linear head type, see model.py for options.
    head_type = config["head_type"]

    # Epsilon value for Adam
    adam_epsilon = config["adam_epsilon"]

    # Optional
    epochs = config.get("num_train_epochs", 1)
    lr_scheduler = config.get("lr_schedule", "constant")
    chat_template = config.get("chat_template", None)
    # Downsize the rank of the classification head.
    linear_head_downsize_factor = config.get("linear_head_downsize_factor", None)
    # Whether to weight the loss. If this is true, it expects that the dataset has a "weight" column.
    weighted_loss = config.get("weighted_loss", False)
    # kwargs for the head init.
    head_config = config.get("head_config", {})
    # If the tokenizer/model does not already have a cls token, this will be used.
    cls_token_if_none = config.get("cls_token_if_none", "<|cls|>")
    # If the tokenizer/model does not already have a pad token, this will be used.
    pad_token_if_none = config.get("pad_token_if_none", "<|pad|>")
    # If using weighted loss, scalar reweight factor
    reweight_scale = config.get("reweight_scale", None)
    proj_name = config.get("proj_name", None)
    init_type = config.get("init_type", "reset_params")
    train_head_only = config.get("train_head_only", False)
    load_train_data_from_disk = config.get("load_train_data_from_disk", False)
    load_val_data_from_disk = config.get("load_val_data_from_disk", False)

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))

    os.makedirs(output_dir, exist_ok=True)

    # define project name
    if not proj_name:
        proj_name = f"{pretrain_model_name.split('/')[1]}_lr{learning_rate}_bs{batch_size}_ep{epochs}"

    print(f"project name: {proj_name}")

    output_path = os.path.join(output_dir, proj_name)

    if args.checkpoint:
        resume_from_checkpoint = args.checkpoint
        print("resuming from checkpoint")
    else:
        resume_from_checkpoint = False

    if not resume_from_checkpoint:
        version = 1
        while os.path.exists(output_path):
            output_path = output_path.replace(f"_{version - 1}", "")
            output_path = output_path + f"_{version}"
            version += 1

    with open(deepspeed_config_path) as fin:
        deepspeed_config = json.load(fin)

    random.seed(42)
    set_seed(42)

    training_args = TrainingArguments(
        output_dir=output_path,
        report_to="wandb",
        run_name=proj_name,
        num_train_epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="no" if args.save_steps == -1 else "steps",
        save_steps=None if args.save_steps == -1 else args.save_steps,
        save_only_model=True,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        ddp_timeout=9999999,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
        eval_steps=args.eval_steps,
        lr_scheduler_type=lr_scheduler,
        logging_dir="./logs",
        fp16=False,
        bf16=True,
        learning_rate=learning_rate,
        adam_epsilon=adam_epsilon,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        do_train=True,
        bf16_full_eval=True,
        save_safetensors=True,
        disable_tqdm=False,
        remove_unused_columns=False,
        deepspeed=deepspeed_config,
        seed=42,
        data_seed=42,
        local_rank=LOCAL_RANK,
    )

    tokenizer = get_tokenizer(
        pretrain_model_name,
        chat_template,
        pad_token_if_none=pad_token_if_none,
        cls_token_if_none=cls_token_if_none,
    )

    data_collator = DataCollator(
        tokenizer, max_length, weight=weighted_loss, reweight_scale=reweight_scale
    )

    train_data = get_dataset(
        train_data_path, "train", from_disk=load_train_data_from_disk
    )

    if not args.no_eval:
        val_data = get_dataset(val_data_path, "train", from_disk=load_val_data_from_disk)

    # with training_args.main_process_first():

    model_list = get_model_list(train_data)

    if not args.no_eval:
        val_model_list = get_model_list(val_data)

        if model_list != val_model_list:
            print("WARNING: Val model list is different, translating...")
            val_data = translate_val_data(val_data, model_list, val_model_list)

    if LOCAL_RANK <= 0:
        # Document the configuration in the output path.
        os.makedirs(output_path, exist_ok=False)

        with open(os.path.join(output_path, "training_config.json"), "w") as fout:
            json.dump(config, fout, indent=1)

        # Save the model list so we know which models this model was trained on. The model list is ALWAYS sorted alphabetically.
        with open(os.path.join(output_path, "model_list.json"), "w") as fout:
            json.dump(model_list, fout, indent=1)

    # Get the model class
    model_cls = get_p2l_model(
        model_type=model_type,
        loss_type=loss_type,
        head_type=head_type,
        init_type=init_type,
    )

    if resume_from_checkpoint:
        print(f"Loading model from checkpoint: {resume_from_checkpoint}")
        model = model_cls.from_pretrained(
            resume_from_checkpoint,
            CLS_id=tokenizer.cls_token_id,
            num_models=len(model_list),
            linear_head_downsize_factor=linear_head_downsize_factor,
        )
    else:
        model = model_cls.from_pretrained(
            pretrain_model_name,
            CLS_id=tokenizer.cls_token_id,
            num_models=len(model_list),
            linear_head_downsize_factor=linear_head_downsize_factor,
        )

    if model.config.vocab_size < len(tokenizer):
        print("WARNING: Resizing Token Embedding")
        model.resize_token_embeddings(len(tokenizer))

    if train_head_only:
        print("Freezing transformer, only training head.")
        model.freeze_transformer()

    trainer = NoShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data.with_format("torch"),
        # eval_dataset=val_data.with_format("torch"),
        data_collator=data_collator,
    )

    print("begin training")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print("saved model and tokenizer")

    if not args.no_eval:
        print("starting eval")
        eval_results = trainer.predict(val_data.with_format("torch"))
        eval_metrics = eval_results.metrics
        eval_predictions = eval_results.predictions
        print(f"Evaluation Results: {eval_metrics}")

        val_set = val_data.add_column("betas", list(eval_predictions[0]))

        if LOCAL_RANK <= 0:
            with open(os.path.join(output_path, "eval_results.json"), "w") as fout:
                json.dump(eval_metrics, fout, indent=1)

            val_dir = os.path.join(output_path, "eval_output.jsonl")
            val_set.to_json(val_dir)
            print(f"saved merged eval results")

    if LOCAL_RANK <= 0:
        if args.push_to_hf:
            api = HfApi()
            repo_id = config.get("repo_id", f"p2el/{proj_name}")
            assert not api.repo_exists(
                repo_id=repo_id, repo_type="model"
            ), "repo already exists"

            api.create_repo(repo_id=repo_id, private=True, repo_type="model")
            api.upload_folder(
                folder_path=output_path,
                repo_id=repo_id,
                repo_type="model",
            )

            print("pushed to hub")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument(
        "--config", type=str, help="path to config file for model training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to checkpoint directory to resume training from",
        default=None,
    )
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        help="True if push directly to huggingface",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=60, help="Number of steps between evaluation."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="If flagged eval will not end at end of training loop.",
    )
    parser.add_argument("--save-steps", type=int, default=-1)

    args = parser.parse_args()

    train_model(args)
