import mlflow
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from model import SentimentModel
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset, random_split

INDEX = 'musics'
es = Elasticsearch("https://localhost:9200/", http_auth=('elastic', '123456'),
                   ca_certs="ca/ca.crt", client_cert="ca/ca.crt", 
                   client_key="ca/ca.key", verify_certs=True)
es.indices.create(index=INDEX, ignore=400)
response = es.search(index=INDEX, body={
    "query": {
        "dis_max": {
            "queries": [
                {"match": {"in_use": False}},
                ]}
            }
    })

features = []
labels = []
if(len(response['hits']['hits']) > 0 ):
    features = [item['_source']['features'] for item in response['hits']['hits']]
    labels = [item['_source']['label'] for item in response['hits']['hits']]

df = pd.DataFrame()
df['features'] = features
df['label'] = labels
mlflow.set_experiment(experiment_id="0")
mlflow.autolog()

features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(pd.factorize(np.array(labels))[0], dtype=torch.long)
dataset = TensorDataset(features_tensor, labels_tensor)
train_size = int(0.8 * len(features))
test_size = len(features) - train_size
_train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataset, _test_dataset = random_split(dataset, [len(features), 0])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_losses = []

model = SentimentModel()
model = model.model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
metric_fn = Accuracy(task="multiclass", num_classes=10).to("cpu")
dataset = mlflow.data.from_pandas(
    df, name="New arriving musics", targets="label"
)

num_epochs = 500
with mlflow.start_run(log_system_metrics=True) as run:
    mlflow.log_input(dataset, context="training")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, label in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs, 1)
            accuracy = metric_fn(predicted, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            mlflow.log_metric("loss", f"{loss:2f}", step=epoch)
            mlflow.log_metric("accuracy", f"{accuracy:2f}", step=epoch)
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        duration = time.time() - start_time
        mlflow.log_metric("duration", duration, step=epoch)

mlflow.end_run()
# torch.save(model.state_dict(), 'model/sentiment_model.pth')
torch.save(model.state_dict(), f"model/sentiment_model_{str(pd.Timestamp.today())}.pth")

