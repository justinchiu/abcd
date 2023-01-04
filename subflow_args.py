import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="14", help="date")

    parser.add_argument("--interact_data", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--true_z", action="store_true")
    parser.add_argument("--kl_weight", default=1.0, type=float)
    parser.add_argument("--kl_no_detach_p", action="store_true")

    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument("--no_save_results", action="store_true")

    parser.add_argument("--num_dialogue_turns", default=0, type=int)
    parser.add_argument("--num_doc_sents", default=0, type=int)

    parser.add_argument("--num_negatives", default=0, type=int, help="only for oracle Z* experiments")

    parser.add_argument("--max_length", default=512, type=int)

    parser.add_argument(
        "--truncate_early",
        action="store_true",
        help="truncate conversations right before first agent action. only allowed during evaluation, since it hurts during training.",
    )

    parser.add_argument("--num_z_samples", default=4, type=int)
    parser.add_argument(
        "--batch_size", "-b", default=1, type=int, help="batch size per gpu."
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="eval batch size per gpu."
    )
    parser.add_argument(
        "--subsampled_batch_size", default=16, type=int, help="batch size per gpu."
    )
    parser.add_argument(
        "--eval_steps",
        default=5000,
        type=int,
        help="number of steps between each evaluation.",
    )
    parser.add_argument(
        "--supervised_examples",
        default=0,
        type=int,
        help="number of fully supervised training examples for q(z|x) and p(x|z). "
        "this should be as small as possible",
    )
    parser.add_argument(
        "--q_warmup_steps",
        default=0,
        type=int,
        help="number of q warmup steps",
    )
    parser.add_argument(
        "--epoch",
        "-epoch",
        default=5,
        type=int,
        help="The number of epochs for fine-tuning.",
    )
    parser.add_argument(
        "--model_dir",
        default="roberta-base",
        type=str,
        help="The directory where the pretrained model will be loaded.",
    )
    parser.add_argument(
        "--answer_model_dir",
        default="facebook/bart-base",
        type=str,
        help="The directory where the pretrained model will be loaded.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--output_model_dir",
        default="./saved_models",
        type=str,
        help="The directory where the pretrained model will be saved.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio in the lr scheduler.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    args = parser.parse_args()
    return args

