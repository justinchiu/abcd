import json
from pathlib import Path
from rich.progress import track

import utils.load
from utils.manual_map import flow_map, subflow_map


# raw_data: Dict[
#   train, dev, test -> List[Example]
# ]
# Example.keys() = convo_id, scenario, original, delexed
# scenario = Dict[
#   personal -> Dict[customer name, email, member_level, phone, username]
#   order -> Dict[
#       street_address,
#       full_address,
#       city,
#       num_products,
#       order_id,
#       packaging,
#       payment_method,
#       purchase_date,
#       state,
#       zip_code,
#       products -> List[Dict[brand, product_type, amount, image]],
#   ]
#   product -> Dict[names, amounts]
#   flow -> Str
#   subflow -> Str
# ]
# original = List[Tuple[speaker, text]]
# delexed = List[Dict[speaker, text, turn_count, targets, candidates]]
#   targets = List[
#       subflow,
#       nextstep in [take_action, retrieve_utterance, end_conversation],
#       action_prediction = button text (30 options),
#       value_filling = slot value (100 options + 155?),
#       utterance ranking = target position of utterance
#           (100 options in candidates),


def get_utterance(example, i):
    utts = [candidates[x] for x in example["delexed"][i]["candidates"]]
    return utts[example["delexed"][i]["targets"][-1]]


def get_subflow_sentences(manual, flow, subflow):
    flow_manual = manual[flow_map[flow]]
    flow_description = flow_manual["description"]

    subflows = flow_manual["subflows"]
    subflow_manual = subflows[subflow_map[subflow]]
    subflow_instructions = subflow_manual["instructions"]
    subflow_actions = subflow_manual["actions"]

    # use mask token for subflow embedding?
    sentences = [f"{subflow} {' '.join(subflow_instructions).lower()}"]
    # use unk tokens for instructions within subflow?
    for action in subflow_actions:
        # main subflow instructions
        sentences.append(f"{action['button'].lower()}: {action['text'].lower()}")
        # TODO: experiment with hierarchical encodings of subtext
        for subtext in action["subtext"]:
            sentences.append(subtext.lower())
    return sentences


def convert_example(
    example,
    cls_token="<s>",
    sep_token="</s>",
    unk_token="<unk>",
    mask_token="<mask>",
):
    """
    EBA example takes the form of text x, z, y.
    x: List[str] = dialogue history
    z: List[str] = subflow sentences
    y: List[str] = language response

    y1: action: bslot, values
    y2: speak: utterance
    y3: end dialogue
    """
    action_texts = [
        f"{speaker.lower()}: {utt.lower()}" for speaker, utt in example["original"]
    ]
    dialogue = [cls_token] + action_texts
    histories = [" ".join(dialogue[:i+1]) for i in range(len(dialogue)-1)]
    x = histories

    # TODO: check how colons get split using tokenizer
    actions = []
    for (speaker, utt), delexed in zip(example["original"], example["delexed"]):
        utt = utt.lower()
        intent, nextstep, slot, values, _ = delexed["targets"]
        if nextstep == "retrieve_utterance":
            actions.append(f"speak: {utt}")
        elif nextstep == "take_action":
            values = ", ".join(values)
            actions.append(f"{slot}: {values}")
        elif nextstep == "end_conversation":
            import pdb; pdb.set_trace()
        else:
            actions.append(None)
    y = actions

    flow = example["scenario"]["flow"]
    subflow = example["scenario"]["subflow"]
    #if subflow in ["status_delivery_date", "status_questions"]:
        #import pdb; pdb.set_trace()
    z = get_subflow_sentences(manual, flow, subflow)

    # filter examples to only non-customer actions
    x,y = list(zip(*[(x,y) for x,y in zip(x,y) if y is not None]))

    return x, y, z


if __name__ == "__main__":
    data_dir = Path("data")
    kb, ontology = utils.load.load_guidelines()
    # 95,288 utterance candidates. where does mapping from turn -> candidates come from?
    with (data_dir / "utterances.json").open("r") as f:
        candidates = json.load(f)
    with (data_dir / "abcd_v1.1.json").open("r") as f:
        raw_data = json.load(f)
    with (data_dir / "guidelines.json").open("r") as f:
        manual = json.load(f)

    save_map = {
        "train": "train",
        "dev": "val",
        "test": "test",
    }

    eba_dir = Path("eba_data")
    eba_dir.mkdir(exist_ok=True)

    for split in ["train","dev","test"]:
        #examples = [convert_example(e) for e in track(raw_data[split])]
        examples = [convert_example(e) for e in raw_data[split]]
        split_str = save_map[split]
        with (eba_dir / f"abcd_{split_str}.json").open("w") as f:
            json.dump(examples, f)
