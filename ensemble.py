import os
import json
import torch
from data import CityFlowNLDatasetInference
from torch.utils.data import DataLoader

data_root = "./extracted_data/"


def process_test(test_query_file, test_tracks_file):
    with open(test_tracks_file) as f:
        test_tracks = json.load(f)
        for k, v in test_tracks.items():
            v["id"] = k
    test_data = list(test_tracks.values())
    test_ds = CityFlowNLDatasetInference(data_root, test_data)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    with open(test_query_file) as f:
        queries = json.load(f)
    return test_dl, queries


def ranking(model_pth, learner, queries, test_dl):
    learner.model.load_state_dict(torch.load(model_pth))
    learner.model.eval()
    with torch.no_grad():
        _, similarity, q_ids = learner.match_track2querry(queries, test_dl)

    similarity = similarity.cpu().numpy()
    assert similarity.shape == (530, 530, 3)
    return similarity, q_ids


def submit(learner, model_folder, test_query_file, test_tracks_file):
    similarity = 0.0
    num = 0.0

    test_dl, queries = process_test(test_query_file, test_tracks_file)
    for model_file in os.listdir(model_folder):
        if model_file.endswith("pth"):
            num += 1
            sim, q_ids = ranking(
                model_folder + "/" + model_file, learner, queries, test_dl
            )
            similarity += sim

    similarity = torch.tensor(similarity / num)
    similarity_mean = similarity.mean(-1).permute(1, 0)

    ids = [item["id"] for item in test_dl.dataset.data]
    results = dict()

    for q_id, simi in zip(q_ids, similarity_mean):
        track_score = {
            id: si for id, si in zip(ids, simi.detach().cpu().numpy().tolist())
        }
        top_tracks = sorted(track_score, key=track_score.get, reverse=True)
        results[q_id] = top_tracks

    return results
