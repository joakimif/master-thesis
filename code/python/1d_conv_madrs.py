from setup_1d_conv import *

setup()

N_FEATURES = 1

segments, labels, num_sensors, input_shape = create_segments_and_labels_madrs(N_FEATURES, segment_length, step)
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

if not model_path:
    if verbose:
        print('Creating model from scratch...')

    model = create_model(segment_length, num_sensors, input_shape, output_classes=4, dropout=dropout)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
    ]

    history = train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.4)
else:
    if verbose:
        print(f'Loading model from {model_path}...')

    model = load_model(model_path)

loss, acc = evaluate(model, X_test, y_test, verbose=verbose)
max_y_test, max_y_pred_test = predict(model, X_test, y_test, verbose=verbose)

if not model_path:
    model.save(f'../models/{identifier}.h5')

    make_confusion_matrix(max_y_test, max_y_pred_test, 
                            output_file=f'../img/confusion_matrix/{identifier}.png', 
                            print_stdout=True, 
                            xticklabels=MADRS_LABLES, 
                            yticklabels=MADRS_LABLES)
else:
    make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)

cleanup()