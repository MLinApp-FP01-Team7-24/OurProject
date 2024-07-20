def plot_loss_moment(history):
    _, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'blue', label='Loss', linewidth=1)
    ax.set_title('Loss over epochs')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(image_dir + 'loss_lstm_vae_' + mode + '.png')

def save_model(model):
    with open(model_dir + 'lstm_vae.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_dir + 'lstm_vae_ckpt')

def load_model():
    lstm_vae_obj = {'Encoder': Encoder, 'Decoder': Decoder, 'Sampling': Sampling}
    with keras.utils.custom_object_scope(lstm_vae_obj):
        with open(model_dir + 'lstm_vae.json', 'r'):
            model = keras.models.model_from_json(model_dir + 'lstm_vae.json')
        model.load_weights(model_dir + 'lstem_vae_ckpt')
    return model