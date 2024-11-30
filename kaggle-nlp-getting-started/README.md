# https://www.kaggle.com/competitions/nlp-getting-started/overview

# We implemented RoBERTa for sentiment analysis 

# During tests we figured out that, ReduceLROnPlateau works fine with our dataset.

# With AdamW and following parameters
    lr=1e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    correct_bias=True

# We hit the score 0.81918