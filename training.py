import torch
from torch import nn, optim
import os

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(state, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch']
        loss = state['loss']
        print(f"Checkpoint loaded: epoch {epoch}, loss {loss}")
        return epoch, loss
    else:
        print("No checkpoint found")
        return 0, float('inf')

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return checkpoints[-1]

def calculate_loss(teacher_logits, student_logits, labels, T, soft_target_loss_weight, ce_loss_weight):
    soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
    soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
    soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T ** 2)

    ce_loss = nn.CrossEntropyLoss()
    label_loss = ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

    loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
    return loss

def perform_train_step(model, optimizer, inputs, labels, attention_mask, decoder_input_ids, teacher_logits, T, soft_target_loss_weight, ce_loss_weight, device):
    optimizer.zero_grad()
    student_outputs = model(input_ids=inputs, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    student_logits = student_outputs.logits
    loss = calculate_loss(teacher_logits, student_logits, labels, T, soft_target_loss_weight, ce_loss_weight)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, optimizer, train_loader, teacher_model, T, soft_target_loss_weight, ce_loss_weight, device):
    model.train()
    teacher_model.eval()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        inputs = batch['input_ids'].to(device)
        labels = batch['summary_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['summary_ids'].to(device)

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=inputs, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            teacher_logits = teacher_outputs.logits

        loss = perform_train_step(
            model, optimizer, inputs, labels, attention_mask, decoder_input_ids,
            teacher_logits, T, soft_target_loss_weight, ce_loss_weight, device
        )
        running_loss += loss

        if step % 1000 == 0:
            print(f"Step {step}/{len(train_loader)}, Loss: {running_loss / (step + 1)}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch Loss: {epoch_loss}")
    return epoch_loss

def train_knowledge_distillation(
    teacher_model, student_model, train_loader, epochs, learning_rate, T,
    soft_target_loss_weight, ce_loss_weight, device, early_stopping, checkpoint_dir
):
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        start_epoch, _ = load_checkpoint(student_model, optimizer, os.path.join(checkpoint_dir, latest_checkpoint), device)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        epoch_loss = train_epoch(
            student_model, optimizer, train_loader, teacher_model,
            T, soft_target_loss_weight, ce_loss_weight, device
        )

        save_checkpoint(student_model, optimizer, epoch, epoch_loss, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    student_model.save_pretrained('checkpoints/student_model')
