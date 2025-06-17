from typing import Dict, List


def split_sequence(sequence: List[Dict[str, str]], label: str):
    """
    Split a sequence of token-label dictionaries into multiple copies, one for each group of
    consecutive tokens with label 'C'. In each copy, set all other labels to 'X'.

    Args:
        sequence: List of dictionaries, each with 'token' and 'label' keys.

    Returns:
        List of sequences, each a copy with one 'C' group preserved and others set to 'X'.
    """
    # Identify groups of consecutive 'C' tokens
    c_groups = []
    current_group = []
    for i, item in enumerate(sequence):
        if item['label'] == label:
            current_group.append(i)
        else:
            if current_group:
                c_groups.append(current_group)
                current_group = []
    if current_group:
        c_groups.append(current_group)

    # Create a copy of the sequence for each group
    result = []
    for group in c_groups:
        # Deep copy the sequence
        new_sequence = [{'token': item['token'], 'label': item['label']} for item in sequence]
        # Set all labels to 'X' except for the current group
        for i in range(len(new_sequence)):
            if i not in group:
                new_sequence[i]['label'] = 'X'
        result.append(new_sequence)

    return result


# Example usage:
if __name__ == "__main__":
    sequence = [
        {'token': 'In', 'label': 'X'},
        {'token': 'contrast', 'label': 'C'},
        {'token': 'to', 'label': 'C'},
        {'token': 'anti-CD3/IL-2-activated', 'label': 'X'},
        {'token': 'LN', 'label': 'X'},
        {'token': 'cells', 'label': 'X'},
        {'token': ',', 'label': 'X'},
        {'token': 'adoptive', 'label': 'C'},
        {'token': 'transfer', 'label': 'X'},
        {'token': 'of', 'label': 'X'},
        {'token': 'freshly', 'label': 'X'},
        {'token': 'isolated', 'label': 'X'},
        {'token': 'tumor-draining', 'label': 'X'},
        {'token': 'LN', 'label': 'X'},
        {'token': 'T', 'label': 'C'},
        {'token': 'cells', 'label': 'X'},
        {'token': 'has', 'label': 'C'},
        {'token': 'no', 'label': 'C'},
        {'token': 'therapeutic', 'label': 'C'},
        {'token': 'activity', 'label': 'X'},
        {'token': '.', 'label': 'X'}
    ]

    sequence = [
        {'token': 'contrast', 'label': 'C'}
    ]

    split_sequences = split_sequence(sequence, "C")
    for i, seq in enumerate(split_sequences):
        print(f"Sequence {i + 1}:")
        print(seq)
        print()