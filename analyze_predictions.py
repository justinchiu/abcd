import torch
import pdb
from pathlib import Path
import json

from subflow_data import get_abcd_dataset

def old_analyze():
    path = (
        "logging/encoder-answer-model-1222-roberta-base-bart-base "
        "lr-2e-05 bs-1 dt-0 ds-0 ml-256 k-32 se-0 qw-0 tz-True kl-0.001 "
        "|step-750.pt"
    )

    out = torch.load(path)

    pdb.set_trace()

def star_all_ws_analyze():
    star = "logging/answer-model-1229-bart-base lr-2e-05 bs-8 dt-0 ds-0 ml-256 k-3 hn-0|step-10000.pt"
    all = "logging/all-answer-model-11-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 |step-5000.pt"
    ws = "logging/ws-encoder-answer-model-1230-roberta-base-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 k-16 tz-False kl-1.0 ndp-True |step-5000.pt"

    star_Z, star_px_z, star_label = torch.load(star)
    all_px_z, all_label = torch.load(all)
    ws_qz, ws_px_z, ws_label = torch.load(ws)


    star_Z = torch.cat(star_Z, 0)
    star_label = torch.cat(star_label, 0)
    all_label = torch.cat(all_label, 0)
    ws_qz = torch.cat(ws_qz, 0).softmax(-1)
    ws_label = torch.cat(ws_label, 0)

    # star already summed
    star_pz = torch.cat(star_px_z, 0)[:,:,128].softmax(-1)
    all_pz = torch.cat(all_px_z, 0).cumsum(-1)[:,:,128].softmax(-1)
    ws_pz = torch.cat(ws_px_z, 0).cumsum(-1)[:,:,128].softmax(-1)

    star_pred = star_pz.argmax(-1)
    all_pred = all_pz.argmax(-1)
    ws_pred = ws_pz.argmax(-1)

    valid_dataset, docs, doc_sents, subflow_map = get_abcd_dataset(
        "dev",
        0,
        0,
        truncate_early=False,
    )

    ds = [" ".join(doc.split()[:32]) for doc in docs]


    disagree = (star_pred != ws_pred)

    star_correct = star_pred == star_label
    ws_incorrect = ws_pred != ws_label

    examples = (star_correct & ws_incorrect).nonzero()[:10,0]

    for idx in examples.tolist():
        print(valid_dataset[idx])
        s = star_pred[idx]
        w = ws_pred[idx]
        l = star_label[idx]

        print("ground truth")
        print(ds[l])
        print("star pred")
        print(ds[s])
        print("ws pred")
        print(ds[w])

        print("label")
        print(ws_label[idx])
        print("ws pz topk")
        print(ws_pz[idx].topk(5))
        print("ws qz topk")
        print(ws_qz[idx].topk(5))
        pdb.set_trace()

def oracle_sent_analyze():
    path = "logging/oracle-sent-model-119-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 |step-5000.pt"
    #path = "logging/oracle-sent-model-119-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-512 s-subflow sk-0 ss-250 sp-0 |step-5000.pt"
    #path = "logging/oracle-sent-model-119-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-512 s-subflow sk-0 ss-250 sp-0 ip-False|step-5000.pt"
    preds, labels, ids = torch.load(path)

    # data
    split = "dev"
    data_dir = Path("data")
    with (data_dir / "abcd_v1.2.json").open("r") as f:
        raw_data = json.load(f)
    with (data_dir / "guidelines.json").open("r") as f:
        manual = json.load(f)
    with (data_dir / "ontology.json").open("r") as f:
        ontology = json.load(f)

    conversations = raw_data[split]

    # annotations files
    datafile = data_dir / "step_annotations.json"
    all_labels = None
    if datafile.exists():
        print("Loading datafile")
        with datafile.open("r") as f:
            all_labels = json.load(f)
    else:
        all_labels = {"dev": {}, "test": {}}

    agent_datafile = data_dir / "agent_step_annotations.json"
    agent_labels = None
    if datafile.exists():
        print("Loading datafile")
        with agent_datafile.open("r") as f:
            agent_labels = json.load(f)
    else:
        agent_labels = {"dev": {}, "test": {}}

    correct = 0
    total = 0
    for i, (pred, label, id) in enumerate(zip(preds, labels, ids)):
        print(i)
        print(id)
        print(label)
        if str(id) not in agent_labels["dev"]:
            continue

        example = [x for x in conversations if x["convo_id"] == id][0]
        dialogue = example["original"]

        this_pred = pred.argmax(0)
        alabel = agent_labels["dev"][str(id)]
        print(this_pred)
        print(alabel)
        for x,y,(speaker, utt) in zip(this_pred.tolist(), alabel, dialogue):
            if y != -1 and speaker == "agent":
                total += 1
                if x == y:
                    correct += 1
    print(f"correct: {correct}")
    print(f"total: {total}")
    import pdb; pdb.set_trace()

def oracle_sent_info_analyze():
    path = "logging/oracle-sent-info-model-21-bart-base lr-2e-05 bs-16 dt-0 ds-0 ml-256 s-subflow sk-0 ss-250 sp-0 ip-True mt-24 mtl-16 msl-128 |step-500.agent.pt"
    preds, labels, ids, fpreds = torch.load(path, map_location=torch.device("cpu"))

    fhats = torch.cat([fpred.argmax(-1) for fpred in fpreds])
    print(fhats.bincount())

    import pdb; pdb.set_trace()

def doc_sent_bad_analyze():
    path = "logging/doc-step-228-roberta-base-bart-base lr-1e-05 bs-16 dt-0 ds-0 ml-256 k-4 ip-True mt-True dta-True |step-1000.pt"
    preds, doc_scores, labels, ids = torch.load(path, map_location=torch.device("cpu"))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    #star_all_ws_analyze()
    #oracle_sent_analyze()
    #oracle_sent_info_analyze()
    doc_sent_bad_analyze()
