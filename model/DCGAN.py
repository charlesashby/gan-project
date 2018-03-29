import tensorflow as tf
from lib.utils import *
from lib.ops import *
from lib.images import *
import time


class DCGAN(object):
    """ DCGAN Implementation """

    def __init__(self):
        # X is of shape ('b', 'sentence_length', 'max_word_length', 'alphabet_size')
        self.hparams = self.get_hparams()
        max_word_length = self.hparams['max_word_length']

        self.output_height = self.hparams['im_height']
        self.output_width = self.hparams['im_width']
        self.batch_size = self.hparams['batch_size']
        self.df_dim = self.hparams['df_dim']
        self.gf_dim = self.hparams['gf_dim']
        self.c_dim = self.hparams['c_dim']
        self.learning_rate = self.hparams['learning_rate']
        self.beta1 = self.hparams['beta1']
        self.epoch = self.hparams['epoch']

        self.z = tf.placeholder('float32', shape=[self.batch_size, self.hparams['z_size']], name='Z')
        self.images = tf.placeholder('float32', shape=[self.batch_size, self.hparams['im_height'],
                                                  self.hparams['im_width'], 3], name='images')


    def build(self):

        # visualization Z variables
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary('d', self.D)
        self.d__sum = histogram_summary('d_', self.D_)
        self.g_sum = histogram_summary('g', self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.g_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.writer = SummaryWriter("./logs", sess.graph)

            done = False
            epoch = 0
            batch = 1
            batch_idxs = 170000 / self.batch_size
            start_time = time.time()
            images_path = glob.glob(PATH + "/*.jpg")
            #self.saver.restore(sess, './checkpoints/dcgan-26001/dcgan')

            while epoch <= self.epoch and not done:
                for mini_batch in iterate_minibatches(self.batch_size, split='train'):
                    batch_z, batch_images = mini_batch
                    # Update D network
                    _, summary_str = sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.images: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, batch)

                    # Update G network
                    _, summary_str = sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, batch)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, batch)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})

                    batch += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, self.epoch, batch, batch_idxs,
                             time.time() - start_time, errD_fake + errD_real, errG))

                    if batch % 100 == 1:
                        #try:
                        samples_z, samples_images = load_data(images_path, 64, 1, split='test')
                        samples = sess.run([self.G], feed_dict={ self.z: samples_z})
                        save_images(samples[0], str(batch))

                    if batch % 1000 == 1:
                        self.saver.save(sess, './checkpoints/dcgan-%s/dcgan' % batch)
                    #except:
                        #    print("one pic error!...")

    def generator(self, z):

        with tf.variable_scope("generator") as scope:

            # Compute the necessary kernel sizes to have
            # an image output of shape:
            # [self.output_height, self.output_width]
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            # with_w: with weights
            self.z_, self.h0_w, self.h0_b = linear(
                    z, 64 * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            # Reshape the projection of z + BN to stabilize for
            # bad initialization
            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, 64 * 8])
            self.g_bn0 = batch_norm(name='g_bn0')
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            self.g_bn1 = batch_norm(name='g_bn1')
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            self.g_bn2 = batch_norm(name='g_bn2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            self.g_bn3 = batch_norm(name='g_bn3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def discriminator(self, image, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            #import pdb; pdb.set_trace()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            self.d_bn1 = batch_norm(name='d_bn1')
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))

            self.d_bn2 = batch_norm(name='d_bn2')
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))

            self.d_bn3 = batch_norm(name='d_bn3')
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def restore(self):
        images_path = glob.glob(PATH + "/*.jpg")
        with tf.Session() as sess:
            self.saver.restore(sess, './checkpoints/dcgan-101/dcgan')
            samples_z, samples_images = load_data(images_path, 64, 1, split='test')
            samples = sess.run([self.G], feed_dict={self.z: samples_z})
            save_images(samples[0], str(1))

    def get_hparams(self):
        """ Get hyper-parameters """
        return {
            # discriminator and generator
            # number of filters for the
            # first deconv or conv
            'df_dim':           64,
            'gf_dim':           64,

            # c_dim: number of channels
            'c_dim':            3,

            # beta param (momentum term) for adam
            'beta1':            0.5,
            'im_height':        64,
            'im_width':         64,
            'z_size':           100,
            'batch_size':       64,
            'epoch':           500,
            'max_word_length':  16,
            'learning_rate':    0.0001,
            'patience':         10000,
        }