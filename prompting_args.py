import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()

    # main options
    parser.add_argument_group("main")
    parser.add_argument("--dataset", choices=["abcd", "flodial", "sgd"], default="abcd")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--embedding_model", default="text-embedding-ada-002")
    parser.add_argument("--log_dir", default="promptlogs")
    parser.add_argument("--num_examples", default=25, type=int)


    # decision options
    parser.add_argument_group("decision")
    parser.add_argument(
        "--docsel",
        choices=["lex", "model", "emb"],
        default="emb",
    )
    parser.add_argument(
        "--stepsel",
        choices=[
            "lex", "model", "emb", "askdial",
            "askturn", "askturnstep",
        ],
        default="lex",
    )
    parser.add_argument(
        "--stepdec",
        choices=["max", "mono", "firstmax", "firstmono"],
        default="max",
    )
    parser.add_argument(
        "--rerank",
        choices=["docscore", "stepscore", "sum"],
        #default="alignscore",
        default="docscore",
    )

    parser.add_argument("--k_docs", type=int, default=3)
    parser.add_argument("--k_steps", type=int, default=3)

    # prompt options
    parser.add_argument_group("stepprompt")
    parser.add_argument(
        "--stepprompt",
        choices=[
            "0s", "4s",
        ],
        default="0s",
    )

    return parser.parse_args()

def get_logpath(args):
    name = "-".join((
        f"{args.dataset}",
        f"n-{args.num_examples}",
        f"ds-{args.docsel}",
        f"ss-{args.stepsel}",
        f"sd-{args.stepdec}",
        f"r-{args.rerank}",
        f"kd-{args.k_docs}",
        f"ks-{args.k_steps}",
        f"sp-{args.stepprompt}",
        f"ds-{args.use_chat}",
    ))
    return Path(args.log_dir) / name
