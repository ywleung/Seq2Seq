import re


class Vocabulary:
    """
    a vocabulary class for Seq2Seq model

    Special tokens:
        <PAD>: padding
        <SOS>: start of sentence
        <EOS>: end of sentence
        <OUT>: out of vocabulary
    """
    VOCAB_FILENAME = 'vocab.tsv'

    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    OUT = '<OUT>'
    special_tokens = [PAD, SOS, EOS, OUT]

    def __init__(self, external_embeddins=None):
        """
        Args:
        :param external_embeddins: word embeddings matrix
        """
        self._word2count = {}
        self._word2index = {}
        self._index2word = {}
        self._compiled = False
        self.external_embeddings = external_embeddins

    def load_word(self, word, word_index, count=1):
        """
        load a word and its integer encoding into vocabulary instance

        :param word: the word to load
        :param word_index: index of the word
        :param count: number of times the word occurs in source corpus
        """
        self._validate_compile(False)

        self._word2count[word] = count
        self._word2index[word] = word_index
        self._index2word[word_index] = word

    def add_words(self, words):
        """
        add a sequence of words to the vocabulary instance

        :param words: sequence of words to add
        """
        self._validate_compile(False)

        for word in words:
            if word in self._word2count:
                self._word2count[word] += 1
            else:
                self._word2count[word] = 1

    def compile(self, vocab_threshold=1, loading=False):
        """
        compile the internal lookup dictionaries that enable words to be integer encoded / decoded.

        :param vocab_threshold: min number of times the word occurs in word corpus to include a word in vocabulary
        :param loading: indicate if the vocabulary is loaded from disk
        """
        self._validate_compile(False)

        if not loading:
            # add special tokens to the lookup dictionaries
            for i, special_token in enumerate(Vocabulary.special_tokens):
                self._word2index[special_token] = i
                self._index2word[i] = special_token

            # add words which meet the threshold in _word2count to the lookup dictionaries
            word_index = len(self._word2index)
            for word, count in sorted(self._word2count.items()):
                if count >= vocab_threshold:
                    self._word2index[word] = word_index
                    self._index2word[word_index] = word
                    word_index += 1
                else:
                    del self._word2count[word]

            # add special tokens to _word2count
            self.add_words(Vocabulary.special_tokens)

        # set compiled to True
        self._compiled = True

    def size(self):
        """
        return number of words of the Vocabulary
        """
        self._validate_compile(True)

        return len(self._word2count)

    def word_exists(self, word):
        """
        check if the word exists in the Vocabulary

        :param word: the word to check
        """
        self._validate_compile(True)

        return word in self._word2index

    def words2indices(self, words):
        """
        encode a sequence of words to a sequence of integers
        """
        return [self.word2index(w) for w in words.split()]

    def word2index(self, word):
        """
        encode a word to an integer
        """
        self._validate_compile(True)

        return self._word2index[word] if word in self._word2index else self.out_index()

    def indices2words(self, words_indices, is_punct_discrete_word=False, capitalize_i=True):
        """
        decode a sequence of indices to a sequence of words

        :param words_indices: a sequence of indices to decode
        :param is_punct_discrete_word: True to output a space before punctuation
                                                        False to output a punctuation just after the last word
        :param capitalize_i:
        """
        words = ''
        for index in words_indices:
            word = self.index2word(index, capitalize_i)
            if is_punct_discrete_word or word not in ['.', '!', '?']:
                words += ' '
            words += word
        words = words.strip()

        return words

    def index2word(self, word_index, capitalize_i=True):
        """
        decode an index to a word

        :param word_index: index to decoder
        :param capitalize_i: indicate if to capitalize character 'i'
        """
        self._validate_compile(True)

        word = self._index2word[word_index]
        if capitalize_i and word == 'i':
            word = 'I'

        return word

    def pad_index(self):
        return self.word2index(Vocabulary.PAD)

    def sos_index(self):
        return self.word2index(Vocabulary.SOS)

    def eos_index(self):
        return self.word2index(Vocabulary.EOS)

    def out_index(self):
        return self.word2index(Vocabulary.OUT)

    def save(self, filepath):
        """
        save the vocabulary to disk

        :param filepath: directory of the file to save to
        """
        total_words = self.size()
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write('\t'.join(['word', 'count']))
            file.write('\n')
            for i in range(total_words):
                word = self._index2word[i]
                count = self._word2count[word]
                file.write('\t'.join([word, str(count)]))
                if i < total_words - 1:
                    file.write('\n')

    def _validate_compile(self, expected_statue):
        """
        validate that the vocabulary is compiled or not based on the needs of the attempted operation

        :param expected_statue: compilation status expected by the attempted operation
        """
        if self._compiled and not expected_statue:
            raise ValueError("This vocabulary instance has already been compiled")
        if not self._compiled and expected_statue:
            raise ValueError("This vocabulary instance has not been compiled yet.")

    @staticmethod
    def load(filepath):
        """
        load vocabulary from disk

        :param filepath: directory of the file to load from
        """
        vocabulary = Vocabulary()

        with open(filepath, 'r', encoding='utf-8') as file:
            for index, line in enumerate(file):
                # skip header line
                if index > 0:
                    word, count = line.split('\t')
                    word_index = index - 1
                    vocabulary.load_word(word, word_index, int(count))

        vocabulary.compile(loading=True)

        return vocabulary

    @staticmethod
    def clean_text(text, max_words=None, normalize_words=True):
        """
        clean text for training and inference

        :param text: text to clean
        :param max_words: max number of words to output
        :param normalize_words: indicate if to normalize words
        """
        text = text.lower()
        text = re.sub(r"'+", "'", text)
        if normalize_words:
            text = re.sub(r"i'm", "i am", text)
            text = re.sub(r"he's", "he is", text)
            text = re.sub(r"she's", "she is", text)
            text = re.sub(r"that's", "that is", text)
            text = re.sub(r"there's", "there is", text)
            text = re.sub(r"what's", "what is", text)
            text = re.sub(r"where's", "where is", text)
            text = re.sub(r"who's", "who is", text)
            text = re.sub(r"how's", "how is", text)
            text = re.sub(r"it's", "it is", text)
            text = re.sub(r"let's", "let us", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"won't", "will not", text)
            text = re.sub(r"shan't", "shall not", text)
            text = re.sub(r"can't", "can not", text)
            text = re.sub(r"cannot", "can not", text)
            text = re.sub(r"n't", " not", text)
            text = re.sub(r"'", "", text)
        else:
            text = re.sub(r"(\W)'", r"\1", text)
            text = re.sub(r"'(\W)", r"\1", text)
        text = re.sub(r"[()\"#/@;:<>{}`+=~|$&*%\[\]_]", "", text)
        text = re.sub(r"[.]+", " . ", text)
        text = re.sub(r"[!]+", " ! ", text)
        text = re.sub(r"[?]+", " ? ", text)
        text = re.sub(r"[,-]+", " ", text)
        text = re.sub(r"[\t]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()

        # Truncate words beyond the limit, if provided. Remove partial sentences from the end if punctuation exists within the limit.
        if max_words is not None:
            text_parts = text.split()
            if len(text_parts) > max_words:
                truncated_text_parts = text_parts[:max_words]
                while len(truncated_text_parts) > 0 and not re.match("[.!?]", truncated_text_parts[-1]):
                    truncated_text_parts.pop(-1)
                if len(truncated_text_parts) == 0:
                    truncated_text_parts = text_parts[:max_words]
                text = " ".join(truncated_text_parts)

        return text

    @staticmethod
    def auto_punctuate(text):
        """
        automatically apply punctuation to text that does not end with any punctuation marks.
        Args:
            text: text to apply punctuation to.
        """
        text = text.strip()
        if not (text.endswith(".") or text.endswith("?") or text.endswith("!") or text.startswith("--")):
            tmp = re.sub(r"'", "", text.lower())
            if (tmp.startswith("who") or tmp.startswith("what") or tmp.startswith("when") or
                    tmp.startswith("where") or tmp.startswith("why") or tmp.startswith("how") or
                    tmp.endswith("who") or tmp.endswith("what") or tmp.endswith("when") or
                    tmp.endswith("where") or tmp.endswith("why") or tmp.endswith("how") or
                    tmp.startswith("are") or tmp.startswith("will") or tmp.startswith("wont") or tmp.startswith("can")):
                text = "{}?".format(text)
            else:
                text = "{}.".format(text)
        return text
