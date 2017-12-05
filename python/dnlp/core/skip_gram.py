# -*- coding: UTF-8 -*-
import pickle
import numpy as np
import math
import tensorflow as tf


class SkipGram(object):
  def __init__(self, src_filename: str, dest_filename: str, batch_size: int = 64, embed_size: int = 100,
               num_sampled: int = 32, steps: int = 50000):
    with open(src_filename, 'rb') as f:
      data = pickle.load(f)
      self.input = data['input']
      self.output = data['output']
      self.dictionary = data['dictionary']
      self.vocab_size = len(self.dictionary)
    self.start = 0
    self.dest_filename = dest_filename
    self.batch_size = batch_size
    self.embed_size = embed_size
    self.num_sampled = num_sampled
    self.size = len(self.input)
    self.steps = steps
    self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))

  def train(self):
    train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

    embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)

    nce_weights = tf.Variable(
      tf.truncated_normal([self.vocab_size, self.embed_size],
                          stddev=1.0 / math.sqrt(self.embed_size)))
    nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

    loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                     num_sampled=self.num_sampled, num_classes=self.vocab_size))
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()

      aver_loss = 0
      for step in range(1,self.steps+1):
        batch_inputs, batch_labels = self.generate_batch()
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        aver_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            aver_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print("Average loss at step ", step, ": ", aver_loss)
          aver_loss = 0
      np.save(self.dest_filename, self.embeddings.eval())

  def generate_batch(self):
    if self.start + self.batch_size > self.size:
      input_batch = self.input[self.start:] + self.input[:self.batch_size + self.start - self.size]
      output_batch = self.output[self.start:] + self.output[:self.batch_size + self.start - self.size]
    else:
      input_batch = self.input[self.start:self.start + self.batch_size]
      output_batch = self.output[self.start:self.start + self.batch_size]

    self.start += self.batch_size
    self.start %= self.size

    return np.array(input_batch, dtype=np.int32), np.expand_dims(np.array(output_batch, dtype=np.int32), 1)
