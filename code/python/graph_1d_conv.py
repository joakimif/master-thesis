from setup_1d_conv import *

N_FEATURES = 1

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
]

if not model_path:
    print('Usage: python3 graph_1d_conv.py [options] --model_path <path_to_models>')
    exit()

output_classes = 2
filename_filter = 'Conv1D'

if madrs:
    create_segments_and_labels = create_segments_and_labels_madrs
    output_classes = 4
    filename_filter = 'Conv1D-MADRS'

loss_list = []
acc_list = []

histories = []
seg_lengths = []

for f in sorted(os.listdir(model_path)):
    if '.h5' in f and filename_filter in f:
        t, ts, seg, step, epochs, batch = f.split('_')
        
        seg = int(seg)
        step = int(step)
        epochs = int(epochs)
        batch = int(batch.replace('.h5', ''))

        segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, seg, step)
        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

        model = create_model(seg, num_sensors, input_shape, output_classes=output_classes)

        h = train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.4)
        loss, acc = model.evaluate(X_test, y_test)

        histories.append(pd.DataFrame(h.history, index=h.epoch))
        seg_lengths.append(seg//60)
        loss_list.append(loss)
        acc_list.append(acc)

        if len(histories) > 2:
            break

historydf = pd.concat(histories, axis=1)

metrics_reported = histories[0].columns
historydf.columns = pd.MultiIndex.from_product([seg_lengths, metrics_reported], names=['hours', 'metric'])

ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ax=ax)
plt.title('Accuracy')

plt.xlabel('Epochs')

plt.tight_layout()
plt.savefig('../img/plot.png')

plt.clf()
plt.plot(seg_lengths, loss_list)
plt.xlabel('Hours')
plt.ylabel('Loss')
plt.savefig('../img/plot_loss_eval.png')

plt.clf()
plt.plot(seg_lengths, acc_list)
plt.xlabel('Hours')
plt.ylabel('Accuracy')
plt.savefig('../img/plot_acc_eval.png')