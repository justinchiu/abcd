import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # main options
    parser.add_argument_group("main")
    parser.add_argument("--dataset", choices=["abcd", "flodial", "sgd"], default="abcd")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--embedding_model", default="text-embedding-ada-002")
    parser.add_argument("--log_name", default="prompting")
    parser.add_argument("--num_examples", default=25, type=int)


    # decision options
    parser.add_argument_group("decision")
    parser.add_argument(
        "--doc_selection",
        choices=["lex", "model", "emb"],
        default="emb",
    )
    parser.add_argument(
        "--step_align",
        choices=["lex", "model", "askdoc", "askturn", "askturnstep"],
        default="lex",
    )
    parser.add_argument(
        "--align_rerank",
        choices=["docscore", "alignscore", "sum"],
        default="alignscore",
    )

    parser.add_argument("--k_docs", type=int, default=3)

    return parser.parse_args()
