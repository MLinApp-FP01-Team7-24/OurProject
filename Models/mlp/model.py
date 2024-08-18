def MLP(params):
    inputs = tf.keras.layers.Input(shape=(X_train.values.shape[1],))
    x = inputs
    for i in range(params.get("mlp_layers")):
        x = tf.keras.layers.Dense(
            units=params.get(f"units_{i}"), activation="relu",
        )(x)
        x = tf.keras.layers.Dropout(rate=params.get("drop_rate"))(x, training=True)
    x = tf.keras.layers.Dropout(rate=0.5)(x, training=True)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    tf_model = tf.keras.Model(inputs=inputs, outputs=x)

    return tf_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_features(window, domain="statistical", frequency=10):
    cfg = tsfel.get_features_by_domain(domain)
    time_series = window.reshape(-1, window.shape[2])
    features = tsfel.time_series_features_extractor(cfg, time_series, fs=frequency, window_size=window.shape[1], verbose=False)
    
    return features

