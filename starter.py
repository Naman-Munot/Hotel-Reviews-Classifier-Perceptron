import sys
import numpy
import time
import tensorflow as tf

USC_EMAIL = 'email@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = 'pass'  # TODO(student): You will be given a password via email.

TRAIN_TIME_MINUTES=11
class DatasetReader(object):
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        lines = []
        output = []
        
        with open(filename, 'r',encoding='UTF-8') as file:
            doc = file.read()
            for line in doc.splitlines():
                lines.append(line.split())
                
        for line in lines:
            parsedLine = []
            temp = [x.rsplit('/',1) for x in line]
            for token in temp:
                if token[0] not in term_index:
                    term_index[token[0]] = len(term_index)
                if token[1] not in tag_index:
                    tag_index[token[1]] = len(tag_index)
                parsedLine.append((term_index[token[0]], tag_index[token[1]]))
            output.append(parsedLine)
        
        return output

    @staticmethod
    def BuildMatrices(dataset):
        term = max([len(x) for x in dataset])
        length = len(dataset)
        terms_mat = numpy.zeros((length, term), dtype=int)
        tags_mat = numpy.zeros((length, term), dtype=int)
        lengths = numpy.zeros((length,), dtype=int)
        
        for a, value in enumerate(dataset):
            lengths[a] = len(value)
            for b, k in enumerate(value):
                terms_mat[a][b] = k[0]
                tags_mat[a][b] = k[1]   
        return (terms_mat, tags_mat, lengths)

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        tag_index = {}
        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)
        
        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                            (train_terms, train_tags, train_lengths),
                            (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)

class SequenceModel(object):
  def __init__(self, max_length=310, num_terms=1000, num_tags=40):
    self.max_length = max_length
    self.num_terms = num_terms
    self.num_tags = num_tags
    self.x = tf.placeholder(tf.int64, [None, max_length], 'X')
    self.lengths = tf.placeholder(tf.int64, [None], 'lengths')
    self.targets = tf.placeholder(tf.int64, [None, max_length], 'targets')
    self.sess=tf.Session()


  def lengths_vector_to_binary_matrix(self, length_vector):
         return tf.sequence_mask(length_vector,self.max_length, dtype=tf.float32)

  def save_model(self, filename):
    """Saves model to a file."""
    pass

  def load_model(self, filename):
    """Loads model from a file."""
    pass

  def build_inference(self):
      em_cap=30
      state_cap=50
      embeddings=tf.get_variable('embeddings',shape=[self.num_terms, em_cap])
      embedd=tf.nn.embedding_lookup(embeddings,self.x)
      lstm_bw = tf.contrib.rnn.BasicLSTMCell(state_cap)
      lstm_fw = tf.contrib.rnn.BasicLSTMCell(state_cap)
      (f_out, b_out), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, embedd, sequence_length=self.lengths, dtype=tf.float32)
      merge = tf.concat((f_out, b_out), axis = -1)
      self.logits = tf.contrib.layers.fully_connected(merge, int(self.num_tags), activation_fn=None)

  def run_inference(self, terms, lengths):
    logits = self.sess.run(self.logits, {self.x: terms, self.lengths: lengths})
    return numpy.argmax(logits, axis=2)

  def build_training(self):
    weights = self.lengths_vector_to_binary_matrix(self.lengths)
    loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,targets=self.targets,weights=weights)
    tf.losses.add_loss(loss)
    opt = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.85, beta2=0.85)  #####
    self.train_opt = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)
    self.sess.run(tf.global_variables_initializer())

  def batch_step(self, terms, tags, lengths):
        self.sess.run(self.train_opt, {self.x: terms, self.targets: tags, self.lengths: lengths})

  def train_epoch(self, terms, tags, lengths, batch_size=32):  ## batch size
    a=terms.shape[0]
    indi=numpy.random.permutation(a)
    for x in range(0,a,batch_size):
        y=min(x+batch_size,a)
        z_slice=terms[indi[x:y]]
        self.batch_step(z_slice,tags[indi[x:y]],lengths[indi[x:y]])
    return True

  def evaluate(self, terms, tags, lengths):
    predicted_tags = self.run_inference(terms, lengths)
    if predicted_tags is None:
      print('Is your run_inference function implented?')
      return 0
    accuracy = numpy.sum(numpy.cumsum(numpy.equal(tags, predicted_tags), axis=1)[numpy.arange(lengths.shape[0]),lengths-1])/numpy.sum(lengths + 0.0)
    print(accuracy)
    return accuracy


def main():
  """This will never be called by us, but you are encouraged to implement it for
  local debugging e.g. to get a good model and good hyper-parameters (learning
  rate, batch size, etc)."""
  # Read dataset.
  reader = DatasetReader
  train_filename = sys.argv[1]
  test_filename = train_filename.replace('_train_', '_dev_')
  term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
  (train_terms, train_tags, train_lengths) = train_data
  (test_terms, test_tags, test_lengths) = test_data

  model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
  model.build_inference()
  model.build_training()
  for j in range(5):
    model.train_epoch(train_terms,train_tags, train_lengths)
    print('Finished epoch %i. Evaluating ...' % (j+1))
    model.evaluate(test_terms, test_tags, test_lengths)

if __name__ == '__main__':
  main()
