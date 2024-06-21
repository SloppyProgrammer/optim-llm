import random

def prune_student_model(student, pruning_rate):
    total_layers = len(student.encoder.block)
    num_layers_to_prune = int(total_layers * pruning_rate)
    print(f"Pruning {num_layers_to_prune} layers out of {total_layers}")

    decoder_indices = list(range(total_layers))
    random.shuffle(decoder_indices)
    prune_indices = decoder_indices[:num_layers_to_prune]
    for idx in sorted(prune_indices, reverse=True):
        del student.decoder.block[idx]

    return student