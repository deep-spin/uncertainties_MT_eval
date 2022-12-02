# uncertainties_MT_eval
Code and data for the paper: [Disentangling Uncertainty in Machine Translation Evaluation](https://arxiv.org/pdf/2204.06546.pdf)


## Quick Installation

We using Python 3.8.

Detailed usage examples and instructions for the COMET metric can be found in the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

To develop locally:
```bash
git clone https://github.com/deep-spin/uncertainties_MT_eval.git
pip install -r requirements.txt
pip install -e .
```

## TL;DR

This repository is en extension of the original COMET metric, providing different options to enhance it with uncertainty predictors. It includes code for **heteroscedastic losses (HTS and KL)**, as well as the option to use the same architecture for **direct uncertainty prediction (DUP)**. 
We used COMET v1.0 as the basis for this extension. 

## Important commands

- To train a new metric use:

    ```bash
    comet-train --cfg config/models/model_config.yaml
    ```

- To use a trained metric of a triplet of a source file <src.txt>, translation file <mt.txt> and reference file <ref.txt> and obtain predictions use:

    ```bash
    comet-score --model <path_to_trained_model> -s src.txt -t mt.txt -r ref.txt
    ```

## Description of configurations and command options

### COMET configuration
To train a plain COMET model on your data without using the uncertainty-related code, use the configuration file :
[uncertainties_MT_eval/configs/models/regression_metric_comet_plain.yaml](../uncertainties_MT_eval/configs/models/regression_metric_comet_plain.yaml)

This model will use an MSE loss and will produce a single output for each segment, corresponding to the predicted **quality score**.

### COMET with MC Dropout configuration

After having (any) trained COMET model you can apply MC Dropout during inference using the ```--mc_dropout``` and specify the desired number *N* of the forward stochastic runs during ```comet-score``` as follows:

```bash
comet-score --model <path_to_trained_model> -s src.txt -t mt.txt -r ref.txt --mc_dropout N
```


This option can be used with models trained using any of the three loss options: hts, kl, mse.

If the option is used with a model trained with the MSE loss, then the model will pgenerateroduce a second output for each segment corresponding to the variance/uncertainty value for each segment's quality score prediction.

If the option is used in combination with any of the two heteroscedastic losses, the model will generate four outputs for each segment in total:
1. The predicted quality score
2. The estimated variance for the quality score
3. The predicted aleatoric uncertainty 
4. The estimated variance of the aleatoric uncertainty 

Then the total uncertainty value for the segment can be calculated as indicated in Eq. 4 in the paper.


>Note that we used N=100 for all experiments in the paper. To reproduce other related works this number might have to be reduced.

### COMET with aleatoric uncertainty predictions

There are two options to train COMET with aleatoric uncertainty prediction. 

1. Heteroscedastic uncertainty (HTS) which can be used with any labelled dataset. It only requires setting the loss to "hts" in the configuration file; see [uncertainties_MT_eval/configs/models/regression_metric_comet_heteroscedastic.yaml](../uncertainties_MT_eval/configs/models/regression_metric_comet_heteroscedastic.yaml) as an example.

2. KL-divergence minimisation based uncertainty (KL). To train a model with the KL setup requires access to labelled data with multiple annotator per segment that provides either (a) multiple human judgements per segment, or (b) the standard deviation of the multiple annotator scores per segment. See file [uncertainties_MT_eval/data/mqm2020/mqm.train.z_score.csv](uncertainties_MT_eval/data/mqm2020/mqm.train.z_score.csv) as an example. 
To train a model on this data set the loss to "kl" in the configuration file. See [uncertainties_MT_eval/configs/models/regression_metric_comet_kl.yaml](../uncertainties_MT_eval/configs/models/regression_metric_comet_kl.yaml)


### COMET-based direct uncertainty prediction (COMET-DUP)

It is possible train a COMET model to predict the uncertainty of a given prediction (casting uncertainty as the error/distance to the human judgement), henceforth referred to as COMET-DUP. 

#### **Training Setup:**

To train a COMET-DUP model it is necessary to:

- Have access to human judgements $q^*$ on a train dataset $\mathcal{D}$  
- Run a MT Evaluation or MT Quality Estimation model to obtain quality predictions  $\hat{q}$ over $\mathcal{D}$
- Calculate $\epsilon = |q^*-\hat{q}|$ for $\mathcal{D}$
- Use $\epsilon$ as the target for the uncertainty predicting COMET, instead of the human quality judgements which is the default target

Provide the training data in a csv file using a column **f1** that holds the values for the predicted quality scores $\hat{q}$ and a column **score** that contains the computed $\epsilon$ (target) for each <src, mt, ref> instance.

#### **Losses**

Upon calculating the above three different losses can be used for the COMET-DUP training:

1. Typical MSE loss: $\mathcal{L}^\mathrm{E}_{\mathrm{ABS}}(\hat{\epsilon}; \epsilon^*) = (\epsilon^* - \hat{\epsilon})^2$\
Specify loss: "mse" in the yaml configuration file to use it
2. MSE loss with squared values: 
   $\mathcal{L}^\mathrm{E}_{\mathrm{SQ}}(\hat{\epsilon}; \epsilon^*) = ((\epsilon^*)^2 - \hat{\epsilon}^2)^2 $
Specify loss: "squared" in the yaml configuration file to use it
3. Heteroschedastic approximation loss:  
$\mathcal{L}^\mathrm{E}_{\mathrm{HTS}}(\hat{\epsilon}; \epsilon^*) = \frac{(\epsilon^*)^2}{2 \hat{\epsilon}^2} + \frac{1}{2}\log(\hat{\epsilon})^2$  
Specify loss: "hts_approx" in the yaml configuration file to use it

#### **Bottleneck**:
COMET-DUP unlike COMET uses a bottleneck layer to incorporate the initial quality predictions $\hat{q}$ as training. You need to specify the the size of the bottleneck layer in the configuration file.  
Recommended value: 256


#### **Full Train Configuration**:
For an example of a configuration file to train COMET-DUP with $\mathcal{L}^\mathrm{E}_{\mathrm{HTS}}$ see the file [uncertainties_MT_eval/configs/models/regression_metric_comet_dup.yaml](../uncertainties_MT_eval/configs/models/regression_metric_comet_dup.yaml)


#### **Inference**

For inference with COMET-DUP use the same inference command (`comet-score`) used for the other COMET models providing a trained COMET-DUP model in the `--model` option. Remember that the output in this case will be uncertainty scores instead of quality scores.

<br>
</br>

***

## Related Publications

- [Better Uncertainty Quantification for Machine Translation Evaluation](https://arxiv.org/pdf/2204.06546.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [IST-Unbabel 2021 Submission for the Quality Estimation Shared Task](https://aclanthology.org/2021.wmt-1.102/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
