from setup_1d_conv import *

setup()

if madrs:
    output_classes = 4
    confusion_matrix_labels = MADRS_LABLES
    create_segments_and_labels = create_segments_and_labels_madrs
else:
    output_classes = 2
    confusion_matrix_labels = CATEGORIES

segments, labels, num_sensors, input_shape = create_segments_and_labels(1, segment_length, step, k_folds=k_folds)

if k_folds > 1:
    models = []
    best_loss = 10000
    best_acc = 0
    i = 0

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    splits = skf.split(segments, labels)

    # print(labels.shape, list(splits))

    for train_index, test_index in splits:
        print(f'Fold: {i+1}/{k_folds}')

        _labels = to_categorical(labels, output_classes)

        X_train, X_test = segments[train_index], segments[test_index]
        y_train, y_test = _labels[train_index], _labels[test_index]

        model = create_model(segment_length, num_sensors, input_shape, output_classes=output_classes, dropout=dropout, verbose=0)
        history = train(model, X_train, y_train, batch_size, epochs, callbacks=[], validation_split=0.4, verbose=1)
        loss, acc = evaluate(model, X_test, y_test, verbose=0) 
        
        models.append((model, history, (loss, acc)))
        
        i+=1

    scores = [x[2] for x in models]
    df = pd.DataFrame(scores, columns=['loss', 'acc'])

    df.to_csv(f'result_{k_folds}_folds_{identifier}.txt')

else:
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

    if not model_path:
        if verbose:
            print('Creating model from scratch...')

        model = create_model(segment_length, num_sensors, input_shape, output_classes=output_classes, dropout=dropout)

        if early_stop:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=early_stop),
            ]
        else:
            callbacks = []

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
                                xticklabels=confusion_matrix_labels, 
                                yticklabels=confusion_matrix_labels)
    else:
        make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)

cleanup()