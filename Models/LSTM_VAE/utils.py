def plot_loss_moment(history):
    _, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'blue', label='Loss', linewidth=1)
    ax.set_title('Loss over epochs')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(image_dir + 'loss_lstm_vae_' + mode + '.png')
