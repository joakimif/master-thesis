from setup_1d_conv import *
from parse_args import segment_length, step

if verbose:
    print('Segment length:', segment_length) 
    print('Step:', step)

N_FEATURES = 1
BATCH_SIZE = 100
EPOCHS = 40

segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, segment_length, step)
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

model = create_model(segment_length, num_sensors, input_shape)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
]

history = train(model, X_train, y_train, BATCH_SIZE, EPOCHS, callbacks, validation_split=0.4)

loss, acc = model.evaluate(X_test, y_test)

if verbose:
    print('Accuracy: {:5.2f}%'.format(100 * acc))
    print('Loss: {:5.2f}%'.format(100 * loss))

max_y_test, max_y_pred_test = predict(model, X_test, y_test)

timestamp = datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")
save_label = f'{segment_length}_{step}_{timestamp}'

model.save(f'models/{save_label}.h5')

make_confusion_matrix(max_y_test, max_y_pred_test, output_file=f'img/confusion_matrix_1d_conv_{save_label}.png', print_stdout=True)