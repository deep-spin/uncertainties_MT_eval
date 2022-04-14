# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command for scoring MT systems.
===============================

optional arguments:
  -h, --help            Show this help message and exit.
  -s SOURCES, --sources SOURCES
                        (required, type: Path_fr)
  -t TRANSLATIONS, --translations TRANSLATIONS
                        (required, type: Path_fr)
  -r REFERENCES, --references REFERENCES
                        (required, type: Path_fr)
  --to_json TO_JSON     (type: Union[bool, str], default: False)
  --model MODEL         (type: Union[str, Path_fr], default: wmt21-large-estimator)
  --batch_size BATCH_SIZE
                        (type: int, default: 32)
  --gpus GPUS           (type: int, default: 1)

"""
import json
from typing import Union

from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from comet.modules import HeteroscedasticLoss, HeteroApproxLoss
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything


def score_command() -> None:
    parser = ArgumentParser(description="Command for scoring MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr, required=True)
    parser.add_argument("-t", "--translations", type=Path_fr, required=True)
    parser.add_argument("-r", "--references", type=Path_fr)
    parser.add_argument("-f", "--features", type=Path_fr, help="Path to additional features for predictor (optional)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--to_json",
        type=Union[bool, str],
        default=False,
        help="Exports results to a json file.",
    )
    parser.add_argument(
        "--model",
        type=Union[str, Path_fr],
        required=False,
        default="wmt20-comet-da",
        #choices=available_metrics.keys(),
        help="COMET model to be used.",
    )
    parser.add_argument(
        "--mc_dropout",
        type=Union[bool, int],
        default=False,
        help="Number of inference runs for each sample in MC Dropout.",
    )
    parser.add_argument(
        "--refless",
        type=bool,
        default=False,
        help="flag for heteroschedastic loss",
    )
    parser.add_argument(
        "--seed_everything",
        help="Prediction seed.",
        type=int,
        default=12,
    )
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    if (cfg.references is None) and ("refless" not in cfg.model) and (not cfg.refless):
        parser.error("{} requires -r/--references.".format(cfg.model))

    model_path = (
        download_model(cfg.model) if cfg.model in available_metrics else cfg.model
    )
    model = load_from_checkpoint(model_path)
    model.eval()

    with open(cfg.sources()) as fp:
        sources = [line.strip() for line in fp.readlines()]

    with open(cfg.translations()) as fp:
        translations = [line.strip() for line in fp.readlines()]

    if cfg.features is not None :
        with open(cfg.features()) as fp:
            features = [(line.strip().split(',')) for line in fp.readlines()]
            features = list(map(list, zip(*features)))
            features = [[float(i) for i in f] for f in features]
            


    if "refless" in cfg.model or cfg.refless:
        if cfg.features is not None :
            data = {"src": sources, "mt": translations}
            for i,f in enumerate(features):
                data['f'+str(i+1)]=f
        else:
            data = {"src": sources, "mt": translations}
    else:
        with open(cfg.references()) as fp:
            references = [line.strip() for line in fp.readlines()]
        if cfg.features is not None :
            data = {"src": sources, "mt": translations, "ref": references}
            for i,f in enumerate(features):
                data['f'+str(i+1)]=f
        else:
            data = {"src": sources, "mt": translations, "ref": references}

    data = [dict(zip(data, t)) for t in zip(*data.values())]
    if cfg.mc_dropout:
        if isinstance(model.loss, HeteroscedasticLoss):
           mean_scores, std_scores, hts_mean, hts_std, sys_score = model.predict(
            data, cfg.batch_size, cfg.gpus, cfg.mc_dropout)
        else:
            mean_scores, std_scores, sys_score = model.predict(
            data, cfg.batch_size, cfg.gpus, cfg.mc_dropout)
        for i, (mean, std, sample) in enumerate(zip(mean_scores, std_scores, data)):
            print("Segment {}\tscore: {:.4f}\tvariance: {:.4f}".format(i, mean, std))
            sample["COMET score"] = mean
            sample["COMET variance"] = std
            if isinstance(model.loss, HeteroscedasticLoss):
                sample["Heteroscedastic score"] = hts_mean
                sample["Heteroscedastic variance"] = hts_std

        print("System score: {:.4f}".format(sys_score))
        if isinstance(cfg.to_json, str):
            with open(cfg.to_json, "w") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)
            print("Predictions saved in: {}.".format(cfg.to_json))

    else:
        if isinstance(model.loss, HeteroscedasticLoss):
            predictions, hts, sys_score = model.predict(data, cfg.batch_size, cfg.gpus) 
        else:
            predictions, sys_score = model.predict(data, cfg.batch_size, cfg.gpus)
        for i, (score, sample) in enumerate(zip(predictions, data)):
            print("Segment {}\tscore: {:.4f}".format(i, score))
            sample["COMET score"] = score
            if isinstance(model.loss, HeteroscedasticLoss):
                sample["Heteroscedastic score"] = hts[i]

        print("System score: {:.4f}".format(sys_score))
        if isinstance(cfg.to_json, str):
            with open(cfg.to_json, "w") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)
            print("Predictions saved in: {}.".format(cfg.to_json))
