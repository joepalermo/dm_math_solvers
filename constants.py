from preprocessing import char2idx

vocab_size = len(char2idx)  # + 2  # +1 for the padding token, which is 0, and +1 for the delimiter newline/start token
question_max_length = 160
answer_max_length = 30