import tensorflow as tf


class DCGAN(object):
    """ DCGAN Implementation """

    def __init__(self, sess):
        # X is of shape ('b', 'sentence_length', 'max_word_length', 'alphabet_size')
        self.hparams = self.get_hparams()
        self.sess = sess
        max_word_length = self.hparams['max_word_length']
        self.Z = tf.placeholder('float32', shape=[None, self.hparams['z_size']], name='Z')
        self.Y = tf.placeholder('float32', shape=[None, self.hparams['im_height'],
                                                  self.hparams['im_width'], 3], name='Y')

    def build(self):


    def get_hparams(self):
        """ Get hyper-parameters """
        return {
            'im_height':        64,
            'im_width':         64,
            'z_size':           100,
            'BATCH_SIZE':       64,
            'EPOCHS':           500,
            'max_word_length':  16,
            'learning_rate':    0.0001,
            'patience':         10000,
        }