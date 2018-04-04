import tensorflow as tf
from lib.utils import *
from lib.ops import *
from lib.images import *
import time


class WGAN(object):
    """ WGAN Implementation """

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
        print('Building WGAN')
        # visualization Z variables
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)


        self.c_loss = tf.reduce_mean(self.D - self.D_)

        # WGAN-GP
        alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
        alpha = alpha_dist.sample((self.batch_size, 1, 1, 1))
        interpolated = self.images + alpha * (self.G - self.images)
        inte_logit = self.discriminator(interpolated, reuse=True)
        gradients = tf.gradients(inte_logit, [interpolated,])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
        gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
        gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
        grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
        self.c_loss += self.hparams['lam'] * gradient_penalty

        self.g_loss = tf.reduce_mean(- self.D_)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)

        self.img_sum = tf.summary.image("img", self.G, max_outputs=10)
        self.theta_g = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.theta_c = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        print('Done building WGAN')


        self.saver = tf.train.Saver()

    def train(self):

        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        opt_g = tf.contrib.layers.optimize_loss(loss=self.g_loss, learning_rate=self.learning_rate,
                                 optimizer=tf.train.AdamOptimizer(beta1=0.5, beta2=0.9),
                                 variables=self.theta_g, global_step=counter_g,
                                 summaries=['gradient_norm'])

        counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_c = tf.contrib.layers.optimize_loss(loss=self.c_loss, learning_rate=self.learning_rate,
                                 optimizer=tf.train.AdamOptimizer(beta1=0.5, beta2=0.9),
                                 variables=self.theta_c, global_step=counter_c,
                                 summaries=['gradient_norm'])
        merged_all = tf.summary.merge_all()
        print('Training')

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.writer = SummaryWriter("./logs", sess.graph)

            done = False
            epoch = 0
            batch = 1
            batch_idxs = 170000 / self.batch_size
            start_time = time.time()
            images_path = glob.glob(PATH + "/*.jpg")
            # self.saver.restore(sess, './checkpoints/dcgan-11001/dcgan')

            for epoch in range(500):
                i = 0
                while i <= batch_idxs and not done:

                    minibatches = iterate_minibatches(self.batch_size, split='train')
                    if i < 25 or i % 500 == 0:
                        citers = 100
                    else:
                        citers = 5

                    for j in range(citers):
                        batch_z, batch_images = next(minibatches)

                        if i % 100 == 99 and j == 0:
                            run_options = tf.RunOptions(
                                trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            _, merged = sess.run([opt_c, merged_all], feed_dict={self.images: batch_images, self.z: batch_z},
                                                 options=run_options, run_metadata=run_metadata)
                            self.writer.add_summary(merged, i)
                            self.writer.add_run_metadata(
                                run_metadata, 'critic_metadata {}'.format(i), i)
                        else:
                            sess.run(opt_c, feed_dict={self.images: batch_images, self.z: batch_z})
                        feed_dict = next(minibatches)
                        if i % 100 == 99:
                            _, merged = sess.run([opt_g, merged_all], feed_dict={self.images: batch_images, self.z: batch_z},
                                                 options=run_options, run_metadata=run_metadata)
                            self.writer.add_summary(merged, i)
                            self.writer.add_run_metadata(
                                run_metadata, 'generator_metadata {}'.format(i), i)
                        else:
                            sess.run(opt_g, feed_dict=feed_dict)
                        if i % 1000 == 999:
                            self.saver.save(sess, './checkpoints/wgan-%s/wgan' % str(i * epoch))

                        if i % 100 == 1:
                            #try:
                            samples_z, samples_images = load_data(images_path, 64, 1, split='test')
                            samples = sess.run([self.G], feed_dict={ self.z: samples_z})
                            save_images(samples[0], str(batch))



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
            self.saver.restore(sess, './checkpoints/wgan-101/dcgan')
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

            # WGAN GP lambda param
            'lam':              10.,
            'im_height':        64,
            'im_width':         64,
            'z_size':           100,
            'batch_size':       64,
            'epoch':           500,
            'max_word_length':  16,
            'learning_rate':    0.00005,
            'patience':         10000,
        }
