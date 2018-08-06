
def makeLayers(input_shape, num_classes):
	import keras
	from keras.models import Sequential, Model
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	from keras import backend as K
	kernel_size = 3
	#Sequential以下で層を積層する
	model = Sequential()
	activator = "relu"
	model.add(Conv2D(16,  kernel_size=kernel_size, activation=activator, input_shape=input_shape)) # Conv2D： 2次元入力をフィルターする畳み込み層
	model.add(Conv2D(32,  kernel_size=kernel_size, activation=activator))
	model.add(MaxPooling2D(pool_size=kernel_size))
	model.add(Dropout(0.25)) # Dropout: 入力にドロップアウトを適用する．ドロップアウトは，訓練時のそれぞれの更新において入力ユニットのrateをランダムに0にセットすることであり，それは過学習を防ぐのを助ける．
	model.add(Conv2D(64,  kernel_size=kernel_size, activation=activator)) # Conv2D： 2次元入力をフィルターする畳み込み層
	model.add(Conv2D(128, kernel_size=kernel_size, activation=activator))
	model.add(MaxPooling2D(pool_size=kernel_size))
	model.add(Dropout(0.25)) # Dropout: 入力にドロップアウトを適用する．ドロップアウトは，訓練時のそれぞれの更新において入力ユニットのrateをランダムに0にセットすることであり，それは過学習を防ぐのを助ける．
	model.add(Flatten())

	model.add(Dense(128, activation=activator)) # Dense： 普通の全結合のニューラルネットワーク
	model.add(Dense(128, activation=activator)) # Dense： 普通の全結合のニューラルネットワーク
	model.add(Dropout(0.25)) # Dropout: 入力にドロップアウトを適用する．ドロップアウトは，訓練時のそれぞれの更新において入力ユニットのrateをランダムに0にセットすることであり，それは過学習を防ぐのを助ける．
	model.add(Dense(64, activation=activator)) # Dense： 普通の全結合のニューラルネットワーク
	model.add(Dense(64, activation=activator)) # Dense： 普通の全結合のニューラルネットワーク
	model.add(Dropout(0.25)) # Dropout: 入力にドロップアウトを適用する．ドロップアウトは，訓練時のそれぞれの更新において入力ユニットのrateをランダムに0にセットすることであり，それは過学習を防ぐのを助ける．
	model.add(Dense(32, activation=activator)) # Dense： 普通の全結合のニューラルネットワーク
	model.add(Dense(16, activation=activator)) # Dense： 普通の全結合のニューラルネットワーク
	model.add(Dense(3, activation=activator)) # Dense： 普通の全結合のニューラルネットワーク
	model.add(Dense(num_classes, activation='softmax')) # Dense： 普通の全結合のニューラルネットワーク

	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adadelta(),
				  metrics=['accuracy'])
	return model