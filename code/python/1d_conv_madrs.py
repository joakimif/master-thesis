from setup_1d_conv import *

N_FEATURES = 1

segments, labels, num_sensors, input_shape = create_segments_and_labels_madrs(N_FEATURES, segment_length, step)
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

if not model_path:
    if verbose:
        print('Creating model from scratch...')

    model = create_model(segment_length, num_sensors, input_shape, output_classes=4)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
    ]

    history = train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.4)
else:
    if verbose:
        print(f'Loading model from {model_path}...')

    model = load_model(model_path)

loss, acc = model.evaluate(X_test, y_test)

if verbose:
    print('Accuracy: {:5.2f}%'.format(100 * acc))
    print('Loss: {:5.2f}%'.format(100 * loss))

max_y_test, max_y_pred_test = predict(model, X_test, y_test)

if not model_path:
    timestamp = datetime.datetime.now().strftime("%m-%d-%YT%H:%M:%S")
    save_label = f'Conv1D-MADRS_{timestamp}_{segment_length}_{step}_{epochs}_{batch_size}'

    model.save(f'../models/{save_label}.h5')

    make_confusion_matrix(max_y_test, max_y_pred_test, 
                            output_file=f'../img/confusion_matrix/{save_label}.png', 
                            print_stdout=True, 
                            xticklabels=MADRS_LABLES, 
                            yticklabels=MADRS_LABLES)
else:
    make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)