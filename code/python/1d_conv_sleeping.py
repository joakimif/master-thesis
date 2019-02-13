from setup_1d_conv import *

N_FEATURES = 1
SEG_LEN = 4*60
STEP = 60

BATCH_SIZE = 50
EPOCHS = 100

segments, labels, num_sensors, input_shape = create_segments_and_labels(n_features, SEG_LEN, STEP, filter_timestamp='night')
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

model = create_model(SEG_LEN, num_sensors, input_shape)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
]

history = train(model, X_train, y_train, BATCH_SIZE, EPOCHS, callbacks)

y_pred_test = model.predict(X_test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

print(classification_report(max_y_test, max_y_pred_test))

make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)