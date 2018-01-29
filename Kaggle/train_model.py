from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def train_model_(model, train_x, train_y, val_x, val_y, file_path):
	
	callbacks = get_callbacks(filepath=file_path, patience=5)
	
	#train our model
	model.fit(train_x, train_y, batch_size=24, epochs=10, verbose=1, validation_data=(val_x, val_y), callbacks=callbacks)
	
	return model