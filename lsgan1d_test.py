from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

def data_generator(samp_len=2000, samp_num=4000):

	data_mat = np.zeros(shape=(samp_num,samp_len))
	data_mat_noise = np.zeros(shape=(samp_num,samp_len))
	data_info = np.zeros(shape=(samp_num,3))

	for i in range(samp_num):
		fc = np.random.randint(low=80000,high=100000)
		n = np.random.randint(7,15)
		t = np.arange(0,n/fc,1E-6)
		tt = np.arange(1E-6,1E-6*(samp_len),1E-6)
		nonl_coef = np.random.uniform(low=0.09,high=0.3)
		sig = 0.5 * (1 - np.cos(2 * np.pi * fc * t / n)) * np.sin(2 * np.pi * fc * t)
		start_pos = np.random.randint(low=0,high=samp_len-len(sig)-1)
		data_mat[i,start_pos:start_pos+len(sig)] = sig[:]
		noise_coef = np.random.uniform(0.1, 0.4)
		theta = np.random.uniform(-np.pi/2,np.pi/2)
		data_mat_noise[i, :] = data_mat[i,:] + noise_coef * np.random.normal(loc=0.0, scale=1.0, size=samp_len) + 0.1*np.sin(2 * np.pi * 50 *tt + theta)
		data_info[i, 0] = fc
		data_info[i, 1] = n
		data_info[i, 2] = nonl_coef

	return data_mat,data_mat_noise,data_info

def noise_generator(samp_len=2000, samp_num=100000):

	data_mat_noise = np.zeros(shape=(samp_num,samp_len))

	for i in range(samp_num):

		nonl_coef = np.random.uniform(low=0.09,high=0.3)
		noise_coef = np.random.uniform(0.1, 0.4)
		data_mat_noise[i, :] = 0.2*np.random.normal(loc=0.0, scale=1.0, size=samp_len) + 0.2*np.random.uniform(-1,1, size=samp_len)

	return data_mat_noise

class LSGAN():
	def __init__(self):
		self.sig_len = 2000
		self.channels = 1
		self.sig_shape = (self.sig_len, self.channels)
		self.latent_dim = 10

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='mse',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generated imgs
		z = Input(shape=(self.latent_dim,))
		signal = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.discriminator(signal)

		# The combined model  (stacked generator and discriminator)
		# Trains generator to fool discriminator
		self.combined = Model(z, valid)
		# (!!!) Optimize w.r.t. MSE loss instead of crossentropy
		self.combined.compile(loss='mse', optimizer=optimizer)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(256, input_dim=self.latent_dim))
		model.add(LeakyReLU(alpha=1))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=1))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=1))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(self.sig_len, activation='tanh'))
		model.add(Reshape(self.sig_shape))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		signal = model(noise)

		return Model(noise, signal)

	def build_discriminator(self):

		model = Sequential()

		model.add(Flatten(input_shape=self.sig_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.5))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.5))
		# (!!!) No softmax
		model.add(Dense(1))
		model.summary()

		signal = Input(shape=self.sig_shape)
		validity = model(signal)

		return Model(signal, validity)

	

	def train(self, epochs, batch_size=128, sample_interval=50):

	# Load the dataset
		X_train = noise_generator()
		# _ , X_train, _ = data_generator()

		# Rescale -1 to 1
		X_train = X_train.astype(np.float32)
		
		X_train = np.reshape(X_train, (len(X_train), 2000, 1))

		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random batch of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			signals = X_train[idx]

			# Sample noise as generator input
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			# noise = np.random.uniform(0, 0.6,(batch_size, self.latent_dim))
			# Generate a batch of new images
			gen_sigs = self.generator.predict(noise)

			# Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(signals, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_sigs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


			# ---------------------
			#  Train Generator
			# ---------------------

			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.sample_images(epoch)
	def sample_images(self, epoch):
		r, c = 5, 2
		noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		gen_sigs = self.generator.predict(noise)
		# print(gen_sigs.shape)

		# Rescale images 0 - 1
		# gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				# axs[i,j].imshow(gen_sigs[cnt, :,:,0], cmap='gray')
				axs[i,j] = plt.subplot(r,c,cnt+1)
				plt.plot(gen_sigs[cnt].reshape(1000))
				# axs[i,j].axis('off')
				cnt += 1
		fig.savefig("signals/signal_%d.png" % epoch)
		plt.close()

if __name__ == '__main__':

	# ori_data,nsy_data,info_data = data_generator()
	# ax = plt.subplot(3, 1, 1)
	# plt.plot(ori_data[1].reshape(1000))
	# ax = plt.subplot(3, 1, 2)
	# plt.plot(nsy_data[1].reshape(1000))
	# plt.show()

	nsy_data = noise_generator()
	plt.plot(nsy_data[1].reshape(2000))
	plt.show()

	gan = LSGAN()
	gan.train(epochs=500000, batch_size=100, sample_interval=1000)  
	gan.generator.save(filepath='lsgan.h5',include_optimizer=False)
	gan.generator.save_weights(filepath='lsgan_weight_8.h5')
