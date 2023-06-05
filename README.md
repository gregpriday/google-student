# Google Student Model Project

## Overview

The Google Student Model project is an internal tool developed with the primary objective of predicting the position of search results in Google. Leveraging the power of machine learning, this project's purpose is to aid in understanding how various factors can influence a web page's search ranking.

The project consists of two primary components: the `app` directory and the `train` directory. The `app` directory contains the main application scripts and models necessary for making predictions. It includes a Flask application (`app.py`), a content model (`content_model.py`), and five trained PyTorch models (`body_model_first.pth`, `body_model_second.pth`, `body_model_third.pth`, `prediction_model.pth`, `title_model.pth`).

On the other hand, the `train` directory consists of data used for training and evaluating the models, stored in two subdirectories (`base` and `name-ideas`), each containing `results.jsonl` and `serps.jsonl`. It also houses the trained models similar to the ones in the `app` directory and a Jupyter notebook (`train.ipynb`) detailing the model training process.

This project is designed for the Google Cloud platform. It leverages Google Cloud Build for building the application and Google Cloud Run for deployment. These technologies enable us to create a continuous integration and continuous deployment (CI/CD) pipeline for effortless updates and improvements to the project.

In the following sections of this document, we will detail three primary phases of this project: training the models, running the model locally, and deploying the application to Google Cloud Run. We will discuss each of these aspects in-depth to facilitate a clear understanding of the operations and expectations for all involved in this project.

## Training

This code is for training a model to predict the search position of a document on Google. The model is called a Google student model.

### I. Importing the Required Libraries

First, the necessary libraries are imported. The code utilizes common libraries such as `numpy`, `torch`, and `wandb` (for logging). `AutoTokenizer`, `AutoModel`, `TrainingArguments`, `Trainer`, and `get_scheduler` are imported from the `transformers` library developed by Hugging Face. 

```python
import numpy as np
import torch
from tqdm.auto import tqdm
import wandb
...
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer, get_scheduler
```

### II. Defining the PositionPredictionModel Class

The main model in this code is `PositionPredictionModel`. It is a custom PyTorch model, which uses transformer models (`cross-encoder/ms-marco-MiniLM-L-4-v2` and `cross-encoder/ms-marco-MiniLM-L-12-v2`) to create sentence embeddings. 

It creates sentence embeddings for the title of the document and for three equally-sized chunks of the body of the document. These embeddings are concatenated and passed through a simple linear model (a feed-forward neural network) to predict the position within 30 bins.

```python
class PositionPredictionModel(nn.Module):

    output_size = 3

    def __init__(self, tokenizer: AutoTokenizer):
        super(PositionPredictionModel, self).__init__()

        self.title_model = AutoModelForSentenceEmbedding(model_name='cross-encoder/ms-marco-MiniLM-L-4-v2')
        self.body_model_first = AutoModelForSentenceEmbedding(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2')
        ...
        self.prediction = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, queries: list, titles: list, bodies: list):
        ...
        return position_likelihood
```

The model also includes `save` and `load` methods to handle model checkpointing and reloading from checkpoints.

**III. Defining the PositionPredictionDataset Class**

The `PositionPredictionDataset` class is used to load and process the data. It loads from `results.jsonl` (containing the search results) and `serps.jsonl` (containing the SERP data) and creates a dataset that can be split into a training set and a test set. 

```python
class PositionPredictionDataset(Dataset):
    def __init__(self, path):
        ...
    def __len__(self):
        return len(self.serps) * 2
    def __getitem__(self, idx):
        ...
```

### IV. Training Loop

Finally, the code contains a training loop where the model parameters are updated for a certain number of epochs. The training loop utilizes the AdamW optimizer and a linear learning rate scheduler. The model is saved after every epoch.

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        queries, titles, bodies, positions = batch
        predictions = model(queries, titles, bodies)
        loss = criterion(predictions, positions.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    model.save('models/base')
    ...
```

### V. Data Format

This script assumes that the data is in the JSON lines (`.jsonl`) format, which is a convenient format for storing structured data that may be processed one record at a time. It works well with Unix-style text processing tools and shell pipelines. It's a great format for log files.

**1. `results.jsonl`:** This file likely contains the search results from Google for different queries. Each line in this file probably represents a single search result and is a JSON object with fields such as:

- `query`: The search query that produced this result.
- `title`: The title of the webpage in the search result.
- `body`: The snippet of the webpage shown in the search result.
- `url`: The URL of the webpage in the search result.
- `position`: The rank or position of the search result in the SERP (search engine results page).

Here's an example of what a single line in `results.jsonl` might look like:

```json
{"query": "best laptops 2023", "title": "Top Laptops of 2023 - PCMag", "body": "Check out our top picks for the best laptops in 2023...", "url": "https://www.pcmag.com/article/363293/the-best-laptops", "position": 1}
```

**2. `serps.jsonl`:** This file likely contains the entire SERP data for different queries. Each line in this file probably represents a single SERP and is a JSON object with fields such as:

- `query`: The search query that produced this SERP.
- `serp`: The entire SERP as a list of search results. Each search result is itself a JSON object with fields such as `title`, `body`, `url`, and `position`.

Here's an example of what a single line in `serps.jsonl` might look like:

```json
{"query": "best laptops 2023", "serp": [{"title": "Top Laptops of 2023 - PCMag", "body": "Check out our top picks for the best laptops in 2023...", "url": "https://www.pcmag.com/article/363293/the-best-laptops", "position": 1}, {"title": "Best Laptops 2023 - TechRadar", "body": "Here's our pick of the best laptops for 2023...", "url": "https://www.techradar.com/news/mobile-computing/laptops/best-laptops-1304361", "position": 2}, ...]}
```

Remember that the actual format of the data depends on how the data was collected and formatted, so the examples above might not match exactly. But they give a rough idea of what the data might look like.

## Running the Model Locally

This section outlines the process of running the `Google Student` model locally on your machine. This is useful for understanding how the model works, making modifications, and testing changes before deploying to the production environment on Google Cloud Run.

### Prerequisites

Make sure you have the following installed:

- Python 3.6 or later
- Flask
- Sentence Transformers
- Torch
- Transformers

You can install the required packages using the command below:

```bash
pip install -r /path/to/requirements.txt
```

### Setting Up

Before starting the Flask server, make sure the `AUTH_TOKEN` environment variable is set to your secret key. This is used to protect your API.

If you want to skip the auth check (only recommended for debugging purposes), you can run the Flask app in debug mode.

### Running the Flask App

Navigate to the `/app` directory and run the Flask app:

```bash
cd /path/to/app
python app.py
```

By default, Flask will start a server on `localhost:5000`. You can change this by providing `host` and `port` parameters to the `app.run()` method in `app.py`.

### Sending Requests

With the server running, you can send a POST request to the `/` endpoint with your data to get predictions. Here is an example of a cURL command:

```bash
curl -X POST -H "Authorization: Bearer YourSecretToken" -H "Content-Type: application/json" -d '{"queries": ["your_queries_here"], "markdown": ["your_markdown_here"]}' http://localhost:5000
```

The data sent with the request should be a JSON object with the following structure:

```json
{
  "queries": ["query_1", "query_2", ...],
  "markdown": ["markdown_1", "markdown_2", ...]
}
```

### Interpreting the Results

The results returned from the server will be a list of scores that represent the position of search results in Google. These scores will be in the same order as the queries and markdown text you provided in your request.

For example, if you send:

```json
{
  "queries": ["query_1", "query_2"],
  "markdown": ["markdown_1", "markdown_2"]
}
```

You might receive the following response:

```json
[0.2, 0.7]
```

This means that `query_1` with `markdown_1` has a score of `0.2` and `query_2` with `markdown_2` has a score of `0.7`.

### Next Steps

Once you've tested your model locally and made any necessary adjustments, you can deploy it to Google Cloud Run. The next section will provide instructions on how to do so.

## Deploying the Model to Google Cloud Run

This section describes the process of deploying our Google Student Model to Google Cloud Run.

### 1. Generate and Set an Authentication Token

We start by generating an AUTH_TOKEN. The AUTH_TOKEN is used in the Flask application for authorizing the requests. You can generate a random AUTH_TOKEN using a tool of your choice. 

Once you have generated the AUTH_TOKEN, you need to set it in the environment variables of the Google Cloud Run instance that will run your application. This can be done through the Cloud Run interface in the GCP Console, under the 'Variables' tab when you're creating or editing a service. 

### 2. Push your Code to the Git Repository

The first step in the deployment process is pushing your local changes to your Git repository. This can be done using the following command:

```
git add .
git commit -m "your commit message"
git push origin main
```

Ensure that you have set up your Git repository to trigger a build in Google Cloud Build upon a push event.

### 3. Cloud Build Triggers

Once your code is pushed to your Git repository, this will automatically trigger a build in Google Cloud Build, assuming you've correctly set up your triggers. 

The Cloud Build service will follow the instructions defined in the `cloudbuild.yaml` file. This file instructs Cloud Build to:

1. Copy models from Google Cloud Storage to the working directory.
2. Build a Docker image using the Dockerfile provided.

In the end, the Docker image is stored in Google Container Registry under the project ID, with the tag `google-student`.

### 4. Deploying to Google Cloud Run

After Cloud Build completes, the Docker image is ready to be deployed. 

1. Navigate to Google Cloud Run in your GCP Console.
2. Click on 'Create Service'.
3. Provide a name for the service.
4. Under 'Container', choose the 'Select' button, then select the Docker image we've just built.
5. Under the 'Capacity' tab, set the Memory Allocated to 8GB and CPU Allocated to 2.
6. Under the 'Variables' tab, set the AUTH_TOKEN environment variable that you generated earlier.
7. Click on 'Create' to deploy your application.

Cloud Run will handle the provisioning and management of the infrastructure for your application, automatically scaling up and down based on incoming traffic.

After successful deployment, Cloud Run provides a URL which can be used to make POST requests to the deployed application.

Please note that the AUTH_TOKEN must be included in the 'Authorization' header of your requests in the format `Bearer YOUR_AUTH_TOKEN`.
