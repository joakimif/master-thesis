from setup_1d_conv import *

N_FEATURES = 1

if not model_path:
    print('Usage: python3 graph_1d_conv.py [options] --model_path <path_to_models>')
    exit()

models = []
datasets = []

for f in os.listdir(model_path):
    if '.h5' in f and 'Conv1D' in f:
        t, ts, seg, step, epochs, batch = f.split('_')

        segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, segment_length, step)

        models.append(load_model(f'{model_path}/{f}'))
        datasets.append(train_test_split(segments, labels, test_size=0.2))


for model, dataset in zip(models, datasets):
    X_train, X_test, y_train, y_test = dataset

    loss, acc = model.evaluate(X_test, y_test)

    if verbose:
        print('Accuracy: {:5.2f}%'.format(100 * acc))
        print('Loss: {:5.2f}%'.format(100 * loss))

    max_y_test, max_y_pred_test = predict(model, X_test, y_test)

    make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)