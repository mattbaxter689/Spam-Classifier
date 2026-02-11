# Spam Classification Model

This repository contains the necessary code to execute a spam classification model, using transfer learning from hugging face models.
The goal for this project is to:

1. Accustom myself to frameworks like pytorch lightning for tuning LLM's on my specific datasets
2. Get more underlying experience with cloud providers

To do so, this project will aim to download a model from hugging face, use lightning to train, mlflow to track the model, and use azure ml as the final end goal
for the code. As time goes on, this documentation will be updated to include and new information, datasets used, decisions in modelling, etc.

# Model and Tokenizing

For this project, I am making use of the `distilbert-base-uncased` model and tokenizer. This model handles textual data, especially conversational-adjacent like emails,
quite well. The idea with this is somewhat simulate what one might see in a production setting. In this case, data could be uploaded or accessed via some S3 or cloud like
storage object. In this case, I make use of my workspace's Blob Storage. You could alternatively set something up that streams data from an S3 bucket, accesses a feature store,
etc. An important note for this project is that I pre-tokenize the data before fitting any sort of model. Tokenizing on the fly would cause massive CPU bottlenecks during model fit,
so pre-tokenizing is the best way to ensure that it is performed efficiently. I have a simple helper script to perform this, and upload it to my workspace's blob storage to access.

# Setup

In order to replicate this project, first ensure that you have an Azure account and the `az` command line tool installed. From there, clone the repo and run the following

```bash
cd infra/terraform/
terraform plan
terraform apply -auto-approve
```

This will make use of terraform to provision all of the resources for this project. When using Azure Machine Learning, we also need to configure the proper compute cluster, compute instances, etc.
The initial options provided are not nearly strong enough for this project, so we need to request a quota increase in order to be able to provision stronger compute. Once you have the resources, you can configure the compute cluster, making sure that `min_nodes=0` and a scale down time. This ensures that after a set amount of time unused, the cluster will spin down, saving on costs.

### Configuring the ENV variables for local runs

To save on time creating ENV variables, I have created a simple shell script to initialize my environment variables. This is simply to save me having to set them every time when running locally. If you are using something like Github Actions you can configure them as secrets.

# Submitting the job

Before we can submit the job, we need to configure the environment for the code to run. Even if we are using local compute, Azure uses a docker container to run the code. I specified it in this case with a `environment/conda.yaml` file, additionally specifying an initial image to use. Another alternative to this is to create your own Dockerfile, and use that image. When creating this, Run the `create_aml_environment.py` file. The image may take several minutes to create. Once configured, the image is ready to use. The caveat with this is that whenever a cold start happens, it will attempt to reinstall the dependencies again. Since we used a conda environment, it would need to rebuild. Instead, we can use the environments ACR image to run the code. You can grab that ACR image link from the environment, and recreate the environment using that link. This will instead reference the image to run the code, severly cutting down time taken to run the code.

Once the environment is successfully created, we can submit the job using the `submit_job.py` file. This will use the image, as well as fetch our pre-transformed data from the blob storage. This will configure the job to run on each specific target.

To make it easy to run these commands, the `Makefile` allows us to easily spin up the environment and job

```bash
make environment
make job
```

# Design Choice

For this project, I wanted to make a more strict choice in terms of design of the model. When using any form of deep learning or machine learning in business
applications, there are certain choices that needs to be made especially regarding optimizing for precision vs recall. In this case, I wanted to have a model that maximized recall during the tuning phase, subjecting the precision to a minimal value. Since misclassifying a normal email as spam could have very severe impact, we want to be able to catch as many spam emails as we can. Given this, we want a minimal acceptable precision, where we can then tune to find the best available recall.

# Project Notes

Going through this project, I have a few comments / issues surrounding this, mostly related to Azure itself.

* If I would create a compute environment again from scratch, I would use a custom docker image rather than what I had done. This would be much easier I feel than creating something, then using the ACR registry to create the environment again, etc. Many manual steps are needed here and just in general is frustrating.
* Azure workspaces making use of mlflow 2.7.1 and using the stages rather than aliasing is slightly frustrating. Understandably, it has been around for some time and I don't see it changing to the new methods any time soon. Part of me does wish it was similar to GCP Vertex where workspaces can choose to spin up mlflow and specify it's version. Regardless, Mlflow being automatically integrated it a large helper
* I am definitely getting more comfortable with torch at this point now which is good. It feels good. I always understood the theory, but the application of it definitely feels good. Torch lightning is also very handy to work with. The integrated metrics and logging is quite nice to have included in this
