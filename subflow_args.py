import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="26", help="date")
    parser.add_argument("--init_from_previous", action="store_true")

    # model choice
    parser.add_argument("--dummy_step", action="store_true",
        help="allow alignment to dummy step, eg no conditioning on a step")

    # objective stuff
    parser.add_argument("--interact_data", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--monotonic_train", action="store_true")

    parser.add_argument("--true_z", action="store_true")
    parser.add_argument("--kl_weight", default=1.0, type=float)
    parser.add_argument("--kl_no_detach_p", action="store_true")

    # logging stuff
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument("--no_save_results", action="store_true")

    # data stuff
    parser.add_argument("--num_dialogue_turns", default=0, type=int)
    parser.add_argument("--num_doc_sents", default=0, type=int)

    parser.add_argument("--num_negatives", default=0, type=int, help="only for oracle Z* experiments")

    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--max_turn_length", default=32, type=int)
    parser.add_argument("--max_step_length", default=128, type=int)
    parser.add_argument("--max_turns", default=128, type=int, help="not used anymore, so set very high. set lower if model ignores history, since this leads to tons of padding.")


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

    # subsample args
    parser.add_argument(
        "--subsampled_batch_size", default=16, type=int, help="batch size per gpu."
    )
    parser.add_argument(
        "--subsample",
        default="subflow",
        choices=["flow", "subflow"],
        help="granularity of subsampling for semisupervised learning",
    )
    parser.add_argument(
        "--subsample_k",
        default=0,
        type=int,
        help="number of supervised examples from each subsampled category. 0 means unsupervised.",
    )

    parser.add_argument(
        "--subsample_passes",
        default=0,
        type=int,
        help="number of passes through supervised examples each time. 0 means unsupervised.",
    )
    parser.add_argument(
        "--subsample_steps",
        default=250,
        type=int,
        help="number of passes through supervised examples each time. 0 means unsupervised.",
    )
    parser.add_argument(
        "--subsample_obj",
        default="joint",
        choices=["joint", "conditional"],
        help="supervised objective",
    )
    parser.add_argument(
        "--subsample_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # / subsample

    parser.add_argument(
        "--eval_steps",
        default=250,
        type=int,
        help="number of steps between each evaluation.",
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

