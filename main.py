import torch
import argparse
from transformers import T5ForConditionalGeneration
from models import prune_student_model
from utils import load_and_preprocess_data
from training import train_knowledge_distillation, EarlyStopping

def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--teacher_model', type=str, default='google/flan-t5-base', help='Teacher model for distillation')
    parser.add_argument('--pruning_rate', type=float, default=0.5, help='Pruning rate for student model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = T5ForConditionalGeneration.from_pretrained(args.teacher_model).to(device)
    student_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to(device)
    student_model = prune_student_model(student_model, args.pruning_rate)

    train_dataset, tokenizer, collator = load_and_preprocess_data(args.teacher_model)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collator)
    early_stopping = EarlyStopping(patience=3, min_delta=0)

    train_knowledge_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        T=2,
        soft_target_loss_weight=0.25,
        ce_loss_weight=0.75,
        device=device,
        early_stopping=early_stopping,
        checkpoint_dir=args.checkpoint_dir
    )

    tokenizer.save_pretrained('./checkpoints/tokenizer')

if __name__ == "__main__":
    main()