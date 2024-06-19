# Word Embeddings with word2vec from Scratch in Python
# https://medium.com/@bradneysmith/word-embeddings-with-word2vec-from-scratch-in-python-eb9326c6ab7c

import matplotlib.pyplot as plt
import numpy as np
import re


class word2vec:
    """ An implementation of the word2vec Skip-Gram algorithm.

    Attributes:
        embedding_dim (int): The number of dimensions in the output word
            embeddings.
        window_size (int): The number of outside words to consider either side
            of a center word.
        epochs (int): The number of epochs to use in training.
        learning_rate (float): The learning rate used for gradient descent to
            iteratively improve the weight matrices. Smaller values (~0.01) are
            generally preferred.
        embeddings (np.array[np.array]): The matrix of static word embeddings
            produced by the model.
        encoded_training_data (np.array[np.array]): A collection of center
            words and their corresponding outside words encoded as one-hot
            vectors.
        losses (list[int]): The loss for each epoch of the model's training.
            Can be used to plot the decrease in the loss over training.
        training_data (list[str]): A list of strings of raw text, which is
            converted to encoded_training_data and used to train the model.
        vec_to_word (dict): A dictionary which maps a string version of the
            one-hot vector to a word string.
        vocab (list): An alphabetised list of all the unique words in the
            training data.
        vocab_size (int): The number of unique words in the training data.
    """

    def __init__(self, embedding_dim, window_size, epochs, learning_rate):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.embeddings = None
        self.encoded_training_data = None
        self.losses = []
        self.training_data = None
        self.vec_to_word = None
        self.vocab = None
        self.vocab_size = None


    def calculate_loss(self, outside_words, y_pred):
        """ Calculate the loss.

        Calculate the loss according to the negative log likelihood function.
        This requires taking the sum of the log P(w_{t+j}|w_t) values (log of
        the softmax outputs, y_pred). To do this, first combine all the one-hot
        encoded outside word vectors into a single vector called
        combined_outside_words. For example, if two outside words have vectors
        [0 1 0 0 0 0] and[0 0 0 0 1 0], the combined_outside_words vector will
        be [0 1 0 0 1 0]. Next multiply this new vector with the softmax
        outputs, y_pred, to obtain the softmax values for the outside words -
        store the result in a vector called components. The components vector
        will have some elements with a value of 0, and so these should be
        removed before taking the elementwise log of the vector. After removing
        and taking the log of the remaining elements, sum the log values to
        give the loss and return the value.

        Args:
            outside_words (np.array[np.arrays]): An array of one-hot encoded
                vectors for each outside word.
            y_pred (np.array): An array of softmax outputs from the network for
              a given center word.

        Returns:
            loss (float): The loss of the network for the input center word.

        """
        # Combine the one-hot outside word vectors into a single vector with 1s
        # for each outside word and 0s for all other words
        combined_outside_words = np.sum(outside_words, axis=0)

        # Multiply by y_pred element-wise to get the components of y_pred that
        # are inside the window
        components = np.multiply(combined_outside_words, y_pred)

        # Find the indexes of the non-zero values in the components vector and
        # remove them to
        # prevent taking log(0) in the next step
        non_zero_idx = np.where(components != 0)[0]
        non_zero_components = components[non_zero_idx]

        # Take the log of the non-zero values
        log_components = np.log(non_zero_components)

        # Sum the log of the
        loss = -np.sum(log_components)

        return loss


    def create_vec_to_word_map(self):
        """ Return a map to convert one-hot vectors back into words."""
        self.vec_to_word = {str(one_hot(word, self.vocab, self.vocab_size)): \
                            word for word in self.vocab}


    def create_vocabulary(self):
        """ Return a sorted list of words by tokenizing training data."""
        all_words = ' '.join(training_data).lower()
        all_words = all_words.replace('.', '')
        all_words = all_words.split(' ')
        self.vocab = list(set(all_words))
        self.vocab.sort()
        self.vocab_size = len(self.vocab)


    def encode_training_data(self):
        """ Encode the center and outside words as one-hot vectors."""

        self.encoded_training_data = []

        for sentence in training_data:

            # Tokenize the sentence
            tokens = re.sub(r'[^\w\s]', '', sentence).lower().split(' ')

            # Encode each center word and its surrounding context words
            for word_pos, word in enumerate(tokens):
                center_word = self.one_hot(word)

                outside_words = []
                for outside_pos in range(word_pos-self.window_size,
                                         word_pos+self.window_size+1):

                    if (outside_pos >= 0) and (outside_pos < len(tokens)) \
                    and (outside_pos != word_pos):

                        outside_words.append(self.one_hot(tokens[outside_pos]))

                self.encoded_training_data.append([center_word, outside_words])


    def one_hot(self, word):
        """ Return a one-hot encoded vector for a word.

        Args:
            word (str): A word from the training data.

        Returns:
            one_hot (np.array): A one-hot encoded vector representation of the
                input word.
        """
        one_hot = [0]*self.vocab_size
        pos = self.vocab.index(word)
        one_hot[pos] = 1
        one_hot = np.array(one_hot)
        return one_hot


    def softmax(self, u):
        """ Return the softmax values for a vector u.

        Args:
            u (np.array): A vector of raw network outputs (logits).

        Returns:
            values (np.array): A vector of softmax values for the input, u.
        """
        values = np.exp(u)/np.sum(np.exp(u))
        return values


    def train(self, training_data):
        """ Take in raw text and produce a matrix of word embeddings.

        From the raw text in training_data, create an alphabetised vocabulary
        of unique words and encode each center word and its corresponding
        outside words as one-hot vectors. Initialise the weights matrices
        W_center and W_outside that store the connections between the layers in
        the network. For each center word in the encoded training data,
        compute a forward, calculate the loss, and compute a backward pass.
        Print the loss for each epoch and repeat until every epoch has been
        completed. Store the final embeddings in the self.embeddings attribute.

        Args:
            training_data (list[str]): A list of strings of raw text, which is
                converted to encoded_training_data and used to train the model.

        Returns:
            None.
        """
        self.create_vocabulary()
        self.encode_training_data()

        # Initialise the connections between layers
        W_center = np.random.rand(self.vocab_size, EMBEDDING_DIM)
        W_outside = np.random.rand(EMBEDDING_DIM, self.vocab_size)

        for epoch_num in range(self.epochs):

            loss = 0

            for x, outside_words in self.encoded_training_data:

                # Forward pass
                h = np.dot(x, W_center)
                u = np.dot(h, W_outside)
                y_pred = self.softmax(u)

                # Calculate the loss
                loss += self.calculate_loss(outside_words, y_pred)

                # Backward pass
                e = np.sum([y_pred - ow for ow in outside_words], axis=0)
                grad_W_outside = np.outer(h, e)
                grad_W_center = np.outer(x, np.dot(W_outside, e))
                W_outside = W_outside - (self.learning_rate * grad_W_outside)
                W_center = W_center - (self.learning_rate * grad_W_center)

            self.losses.append(loss)
            print(f'Epoch: {epoch_num+1}    Loss: {loss}')

        self.embeddings = W_center


EMBEDDING_DIM = 3
WINDOW_SIZE = 2
EPOCHS = 1000
LEARNING_RATE = 0.01

# Create training data
training_data = ['The dog chased the cat around the garden.',
                 'The cat drank some water.',
                 'The dog ate some food.',
                 'The cat ate the mouse.']

# Instantiate the model
w2v = word2vec(embedding_dim = EMBEDDING_DIM,
               window_size = WINDOW_SIZE,
               epochs = EPOCHS,
               learning_rate = LEARNING_RATE)

# Train the model
w2v.train(training_data)




