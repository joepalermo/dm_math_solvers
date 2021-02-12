from sklearn.metrics import mean_absolute_error
import random
import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, case, sequence, target):
        self.case = case
        self.sequence = sequence
        self.target = target

    def __len__(self):
        return len(self.case)

    def __getitem__(self, idx):
        return self.case[idx], self.sequence[idx], self.target[idx]


def generate_dataset(num_unique_samples, num_cases, num_safe_distractors, num_kill_distractors, sequence_length):
    '''
    samples can have target 0 if they have either:
    1) the wrong number of 0s to match the case
    2) distractor symbols
    otherwise
    '''
    targets = []
    samples = []
    unique_set = set()
    start_token = num_safe_distractors + num_kill_distractors + 1
    padding_token = num_safe_distractors + num_kill_distractors + 2
    while len(samples) < num_unique_samples:
        sequence = []
        # pick case and target
        case = random.randint(0, num_cases-1)
        target = random.randint(0, 1)
        # generate sequence
        if target == 1:
            # case: count correct + safe distractors
            sequence.extend([0] * case)
            sampled_num_distractors = random.randint(0, sequence_length - len(sequence) - 1)
            distractors = [random.randint(1, num_safe_distractors) for _ in range(sampled_num_distractors)]
            sequence.extend(distractors)
        elif random.random() < 0.5:
            # case: count incorrect + safe distractors
            wrong_num_cases = case
            while wrong_num_cases == case:
                wrong_num_cases = random.randint(0, num_cases-1)
            sequence.extend([0] * wrong_num_cases)
            sampled_num_distractors = random.randint(0, sequence_length - len(sequence) - 1)
            distractors = [random.randint(1, num_safe_distractors) for _ in range(sampled_num_distractors)]
            sequence.extend(distractors)
        else:
            # kill distractors
            if random.random() < 0.5:
                # case: count correct + kill distractors
                sequence.extend([0] * case)
            else:
                # case: count incorrect + kill distractors
                wrong_num_cases = case
                while wrong_num_cases == case:
                    wrong_num_cases = random.randint(0, num_cases - 1)
                sequence.extend([0] * wrong_num_cases)
            # add kill distractors
            sampled_num_distractors = random.randint(0, sequence_length - len(sequence) - 1)
            distractors = [random.randint(num_safe_distractors + 1, num_safe_distractors + num_kill_distractors)
                           for _ in range(sampled_num_distractors)]
            sequence.extend(distractors)
        # shuffle
        random.shuffle(sequence)
        # add padding
        padding = [padding_token for _ in range(sequence_length - len(sequence))]
        sequence.extend(padding)
        # add start token
        sequence = [start_token] + sequence
        # put sample together
        sample = (case, sequence, [target])
        hashable_sample = f'{case}_{",".join([str(x) for x in sequence])}_{target}'
        if hashable_sample not in unique_set:
            unique_set.add(hashable_sample)
            samples.append(sample)
            targets.append(target)
    mean_target = sum(targets)/len(targets)
    print('mean_target:', mean_target)
    baseline = [mean_target for _ in targets]
    print('baseline MAE: ', mean_absolute_error(baseline, targets))
    return samples