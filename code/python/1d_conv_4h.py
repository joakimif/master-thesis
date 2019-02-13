from setup_1d_conv import *

N_FEATURES = 1
SEG_LEN = 4*60
STEP = 60

BATCH_SIZE = 100
EPOCHS = 40

segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, SEG_LEN, STEP)
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

model = create_model(SEG_LEN, num_sensors, input_shape)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
]

history = train(model, X_train, y_train, BATCH_SIZE, EPOCHS, callbacks, validation_split=0.3)

max_y_test, max_y_pred_test = predict(model, X_test, y_test)

make_confusion_matrix(max_y_test, max_y_pred_test, save_image_location='img/4h/confusion_matrix.png', print_stdout=True)