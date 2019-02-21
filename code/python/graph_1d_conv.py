from setup_1d_conv import *

N_FEATURES = 1

if not model_path:
    print('Usage: python3 graph_1d_conv.py [options] --model_path <path_to_models>')
    exit()

results = []

for f in os.listdir(model_path):
    if '.h5' in f and 'Conv1D' in f:
        t, ts, seg, step, epochs, batch = f.split('_')
        batch = batch.replace('h5', '')

        segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, int(seg), int(step))

        model = load_model(f'{model_path}/{f}')

        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

        loss, acc = model.evaluate(X_test, y_test)
        max_y_test, max_y_pred_test = predict(model, X_test, y_test)

        results.append(((t, ts, seg, step, epochs, batch), (loss, acc), (max_y_test, max_y_pred_test)))

        # print('Accuracy: {:5.2f}%'.format(100 * acc))
        # print('Loss: {:5.2f}%'.format(100 * loss))
        # make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)

for result in results:
    print(result)