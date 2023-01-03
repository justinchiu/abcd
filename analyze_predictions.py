import torch
import pdb

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

    valid_dataset, docs, subflow_map = get_abcd_dataset(
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

star_all_ws_analyze()
