from setup_1d_conv import *

N_FEATURES = 1

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
]

if not model_path:
    print('Usage: python3 graph_1d_conv.py [options] --model_path <path_to_models>')
    exit()

histories = []
seg_lengths = []

for f in reversed(os.listdir(model_path)):
    if '.h5' in f and 'Conv1D' in f:
        t, ts, seg, step, epochs, batch = f.split('_')
        
        seg = int(seg)
        step = int(step)
        epochs = int(epochs)
        batch = int(batch.replace('.h5', ''))

        segments, labels, num_sensors, input_shape = create_segments_and_labels(N_FEATURES, seg, step)
        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

        model = create_model(seg, num_sensors, input_shape)
        h = train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.4)

        df = pd.DataFrame(h.history, index=h.epoch)
        
        histories.append(df)
        seg_lengths.append(seg)

        #if len(histories) > 1:
        #    break

historydf = pd.concat(histories, axis=1)

metrics_reported = histories[0].columns
historydf.columns = pd.MultiIndex.from_product([seg_lengths, metrics_reported], names=['segment length', 'metric'])

ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Accuracy')

plt.xlabel('Epochs')

plt.tight_layout()
plt.savefig('../img/plot.png')

