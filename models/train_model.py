def train_model(model, x_train, y_train, epochs=5, batch_size=32, validation_split=0.1):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
