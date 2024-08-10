from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class LayerwiseLearningRateOptimizer:
    def __init__(self, model, lr_bert=2e-5, lr_task_heads=5e-5, lr_decay=0.95):
        self.lr_bert = lr_bert
        self.lr_task_heads = lr_task_heads
        self.lr_decay = lr_decay

        # Grouping parameters
        bert_params = list(model.bert.named_parameters())
        task_head_params = list(model.task_A_classifier.parameters()) + \
                           list(model.task_B_classifier.parameters()) + \
                           list(model.linear.parameters())

        # Creating parameter groups with layer-wise learning rates
        self.optimizer_grouped_parameters = []

        # BERT layers
        for layer in range(len(bert_params) - 1, -1, -1):
            layer_params = bert_params[layer][1]
            layer_lr = self.lr_bert * (self.lr_decay ** (len(bert_params) - 1 - layer))
            self.optimizer_grouped_parameters.append({
                "params": layer_params,
                "lr": layer_lr
            })

        # Task-specific heads
        self.optimizer_grouped_parameters.append({
            "params": task_head_params,
            "lr": self.lr_task_heads
        })

        # Create the optimizer
        self.optimizer = AdamW(self.optimizer_grouped_parameters)

    def get_optimizer(self):
        return self.optimizer

#Usage
model = MultiTaskSentenceTransformer()
optimizer_wrapper = LayerwiseLearningRateOptimizer(model)
optimizer = optimizer_wrapper.get_optimizer()

#  Creating a learning rate scheduleer
num_training_steps = 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

#training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        loss = compute_loss(model, batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Zero gradients
        optimizer.zero_grad()