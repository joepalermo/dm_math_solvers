import random
import pickle
# hparams
n = 10000
vocab_size = 1 #number of unique characters
max_count = 5 # largest possible target

#init stuff
data = list()
vocab = [chr(i) for i in range(97, 97 + vocab_size) ]
for i in range(n):
    number_of_each = [random.randint(0,max_count) for _ in range(vocab_size)]
    unshuffled = ''
    for character, n_character in zip(vocab, number_of_each):
        unshuffled += character * n_character
    shuffled = ''.join(random.sample(unshuffled, len(unshuffled)))

    char_to_count_idx = random.randint(0, vocab_size-1)
    char_to_count = vocab[char_to_count_idx]
    target = str(number_of_each[char_to_count_idx])
    input = f"count the number of {char_to_count}: {shuffled}"
    data.append((input, target))

# pickle data
with open('../../preprocessing/counting_dataset.pkl', 'wb') as f:
    pickle.dump(data, f)