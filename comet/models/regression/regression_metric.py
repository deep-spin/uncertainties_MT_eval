# -*- coding: utf-8 -*-
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

r"""
RegressionMetric
========================
    Regression Metric that learns to predict a quality assessment by looking
    at source, translation and reference.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from comet.models.base import CometModel
from comet.modules import FeedForward, Bottleneck
from torchmetrics import MetricCollection, PearsonCorrcoef, SpearmanCorrcoef
from transformers import AdamW
import random

class RegressionMetric(CometModel):
    """RegressionMetric:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the embeddings frozen during training.
    :param keep_encoder_frozen: freezes entire encoder.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        keep_encoder_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        hidden_sizes_bottleneck: List[int] = [3072, 1024],
        hidden_sizes: List[int] = [256],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
        loss: Optional[str]="mse",
        data_portion: Optional[float] = 1.0,
        feature_size: Optional[int] = 0
    ) -> None:
        super().__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            keep_encoder_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "regression_metric",
        )
        self.save_hyperparameters()

        if self.hparams.hidden_sizes_bottleneck[0]>0:
            self.bottleneck = Bottleneck(
                in_dim=self.encoder.output_units * 6 ,
                hidden_sizes = [self.hparams.hidden_sizes[0],self.hparams.hidden_sizes_bottleneck[-1]],
                activations=self.hparams.activations,
                dropout=self.hparams.dropout,
            )

            self.estimator = FeedForward(
                in_dim=self.hparams.hidden_sizes_bottleneck[-1] + self.hparams.feature_size,
                out_dim = 2 if self.hparams.loss in ["var", "hts"] else 1,
                hidden_sizes=[self.hparams.hidden_sizes[-1]],
                activations=self.hparams.activations,
                dropout=self.hparams.dropout,
                final_activation=self.hparams.final_activation,
            )
        else:
            self.estimator = FeedForward(
                in_dim=self.encoder.output_units * 6,
                out_dim = 2 if self.hparams.loss in ["var", "hts"] else 1,
                hidden_sizes=self.hparams.hidden_sizes,
                activations=self.hparams.activations,
                dropout=self.hparams.dropout,
                final_activation=self.hparams.final_activation,
            )

    def init_metrics(self):
        metrics = MetricCollection(
            {"spearman": SpearmanCorrcoef(), "pearson": PearsonCorrcoef()}
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets the optimizers to be used during training."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.estimator.parameters() , "lr": self.hparams.learning_rate}
        ]
        if self.hparams.hidden_sizes_bottleneck[0]>0:
            bott_layers_parameters = [
                {"params": self.bottleneck.parameters() , "lr": self.hparams.learning_rate}
            ]
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            if self.hparams.hidden_sizes_bottleneck[0]>0:
                params = layer_parameters + top_layers_parameters + bott_layers_parameters + layerwise_attn_params
            else:
                params = layer_parameters + top_layers_parameters + layerwise_attn_params
        else:
            if self.hparams.hidden_sizes_bottleneck[0]>0:
                params = layer_parameters + top_layers_parameters + bott_layers_parameters
            else:
                params = layer_parameters + top_layers_parameters

        optimizer = AdamW(
            params,
            lr=self.hparams.learning_rate,
            correct_bias=True,
        )
        # scheduler = self._build_scheduler(optimizer)
        return [optimizer], []

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False, data_portion: float = 1.0,
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        #print(sample[0])
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = self.encoder.prepare_sample(sample["ref"])
        
        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        if self.hparams.feature_size>0:
            feats = []
            for feat in sample:
                if feat.startswith("f"):
                    feats.append(sample[feat])
            #print(len(feats))
            feature_tensor = torch.as_tensor(feats, dtype=torch.float)
            #print(feature_tensor.shape)
            #print('------------------')
            features = {"custom_features": feature_tensor.T}

            
        else:
            features = {"custom_features": torch.Tensor()}

        inputs = {**src_inputs, **mt_inputs, **ref_inputs, **features}
        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        ref_input_ids: torch.tensor,
        ref_attention_mask: torch.tensor,
        custom_features: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        mt_sentemb = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)
        ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
            dim=1,
        )
        if self.hparams.feature_size>0 and self.hparams.hidden_sizes_bottleneck[0]>0:
            bottleneck = self.bottleneck(embedded_sequences) 
            seq_feats = torch.cat((bottleneck,custom_features),dim=1)
            score = self.estimator(seq_feats)
        elif self.hparams.feature_size==0 and self.hparams.hidden_sizes_bottleneck[0]>0:
            bottleneck = self.bottleneck(embedded_sequences)
            score = self.estimator(bottleneck)
        else:
            #bottleneck = self.bottleneck(embedded_sequences)
            score = self.estimator(embedded_sequences)
        if self.hparams.loss in ["var","hts"]:
            return {"score": score[:,0], "variance": score[:,1]}

        return {"score": score}

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        feats=[]
        df = pd.read_csv(path)
        flen = self.hparams.feature_size
        columns = ["src", "mt", "ref", "score"]
        for i in range(flen):
            fstring='f'+str(i+1)
            print('feature added: '+str(fstring))
            columns.append(fstring)
            feats.append(fstring)
        df = df[columns]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].astype(str)
        df["score"] = df["score"].astype(float)
        for feat in feats:
            df[feat] = df[feat].astype(float)
        return df.to_dict("records")
