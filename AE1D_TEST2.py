#jazzy

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Convolution2D,Conv1D,MaxPooling1D,Convolution1D,UpSampling1D
from keras.models import Model
from keras.utils import plot_model

def fft_show(ori_func, ft, sampling_period = 1E-7):
    n = len(ori_func)
    interval = sampling_period / n
    frequency = np.arange(n / 2) / (n * interval)
    nfft = abs(ft[range(int(n / 2))] / n )
    plt.plot(frequency, nfft, 'red')

def plot_train(model_name):
    fig = plt.figure()
    plt.plot(model_name.history['loss'], label='training loss')
    plt.plot(model_name.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('CAE' + 'loss.png')

samp_num = 4000
samp_len = 1000
data_mat = np.zeros(shape=(samp_num,samp_len))
data_mat_noise = np.zeros(shape=(samp_num,samp_len))
data_info = np.zeros(shape=(samp_num,3))

for i in range(samp_num):
    fc = np.random.randint(low=60000,high=80000)
    # fc = 70000
    fn = fc*2
    n = np.random.randint(7,15)
    nn = n*2
    t = np.arange(0,n/fc,5E-7)
    nonl_coef = np.random.uniform(low=0.09,high=0.3)
    #sig = 0.5*(1-np.cos(2*np.pi*fc*t/n))*np.sin(2*np.pi*fc*t)+nonl_coef*(1-np.cos(2*np.pi*fn*t/nn))*np.sin(2*np.pi*fn*t)
    sig = 0.5 * (1 - np.cos(2 * np.pi * fc * t / n)) * np.sin(2 * np.pi * fc * t)
    start_pos = np.random.randint(low=0,high=samp_len-len(sig)-1)
    #print(i,len(sig),samp_len-len(sig)-1)
    data_mat[i,start_pos:start_pos+len(sig)] = sig[:]
    noise_coef = np.random.uniform(0.1, 0.4)
    data_mat_noise[i, :] = data_mat[i,:] + noise_coef * np.random.normal(loc=0.0, scale=1.0, size=samp_len)
    data_info[i, 0] = fc
    data_info[i, 1] = n
    data_info[i, 2] = nonl_coef

x_train = data_mat_noise[:int(0.8*samp_num),:]
x_target = data_mat[:int(0.8*samp_num),:]
x_test = data_mat_noise[int(0.8 * samp_num):, :]
x_test_target = data_mat[int(0.8 * samp_num):, :]
print(x_train.shape)
x_train = np.reshape(x_train, (len(x_train), samp_len, 1))
x_test = np.reshape(x_test, (len(x_test), samp_len, 1))
x_target = np.reshape(x_target, (len(x_target), samp_len, 1))
x_test_target = np.reshape(x_test_target, (len(x_test_target), samp_len, 1))
print(x_test.shape)
# plt.contourf(data_mat_noise)
# plt.show()
# plt.plot(data_mat_noise[300,:])
# plt.show()
input_img = Input(shape=(samp_len, 1))

x = Conv1D(filters = 20, kernel_size = 20, activation='relu', padding='same')(input_img)
x = MaxPooling1D(pool_size = 2, padding='same')(x)

x = Conv1D(filters = 15, kernel_size = 20, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size = 2, padding='same')(x)

x = Conv1D(filters = 15, kernel_size = 20, activation='relu', padding='same')(x)
encoded = MaxPooling1D(pool_size = 2, padding='same')(x)

x = Conv1D(15, 20, activation='relu', padding='same')(encoded)
x = UpSampling1D(2)(x)

x = Conv1D(15, 20, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)

x = Conv1D(20, 20, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 5, activation='tanh', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


history = autoencoder.fit(x_train, x_target,
                epochs=15,
                batch_size=200,
                shuffle=True,
                validation_data=(x_test, x_test_target)
               )

autoencoder.save(filepath='convae1d_6.h5',include_optimizer=False)
autoencoder.save_weights(filepath='convae1d_weight_6.h5')
#plot_model(autoencoder, to_file= 'modelcons' + '.png')

decoded_imgs = autoencoder.predict(x_test)

n = 10

print(history.history.keys())
plot_train(history)


plt.figure(figsize=(30, 5))
for i in range(n):
    ax = plt.subplot(5, n, i + 1)
    # print(data_info[int(0.8 * samp_num) + i, 0]
    #         , data_info[int(0.8 * samp_num) + i, 1]
    #         , data_info[int(0.8 * samp_num) + i, 2]
    #     )
    plt.plot(x_test[i].reshape(samp_len))


    ax = plt.subplot(5, n, i+1+n)
    plt.plot(decoded_imgs[i].reshape(samp_len))

    # ax = plt.subplot(5, n, i + 1 + 2*n)
    # plt.plot(x_test_target[i].reshape(1000))

    ax = plt.subplot(5, n, i + 1 + 2 * n)
    fft_res = np.fft.fft(decoded_imgs[i].reshape(samp_len))
    fft_show(decoded_imgs[i].reshape(samp_len),fft_res)

    ax = plt.subplot(5, n, i + 1 + 3 * n)
    fft_res = np.fft.fft(x_test[i].reshape(samp_len))
    fft_show(x_test[i].reshape(samp_len), fft_res)

plt.show()
