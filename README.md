# Spam Classification Model

This repository contains the necessary code to execute a spam classification model, using transfer learning from hugging face models.
The goal for this project is to:
1. Accustom myself to frameworks like pytorch lightning for tuning LLM's on my specific datasets
2. Get more underlying experience with cloud providers

To do so, this project will aim to download a model from hugging face, use lightning to train, mlflow to track the model, and use azure ml as the final end goal
for the code. As time goes on, this documentation will be updated to include and new information, datasets used, decisions in modelling, etc. 

# Data and Loading
The dataset for this model comes from the Enron email dataset. This dataset contains ~30,000 emails that are a mixture of spam/not spam. In combination with this
we use the `distilbert-base-uncased` model and associated tokenizer from Hugging Face. Since this tokenizer handles much of the text processing itself (since it considers
typical text cleaning steps as valuable and part of it's pipeline), we can transform the data without worry. If there was a transformation we wanted to perform, we could do something like
remove HTML tags from the email, but I have left that out for now.
