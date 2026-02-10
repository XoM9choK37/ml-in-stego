import numpy as np
import tensorflow as tf
import os
import json

class PeriodicSaveConfig(tf.keras.callbacks.Callback):
    def __init__(self, dirpath='model_checkpoints', period=1,
                 weights_name_template='weights_epoch_{epoch:04d}.h5',
                 model_json_name='model_config.json',
                 training_info_template='training_info_epoch_{epoch:04d}.json',
                 save_model_json_every=None,
                 verbose=1):
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.dirpath = dirpath
        self.period = int(period)
        self.weights_name_template = weights_name_template
        self.model_json_name = model_json_name
        self.training_info_template = training_info_template
        self.save_model_json_every = save_model_json_every
        self.verbose = verbose
        os.makedirs(self.dirpath, exist_ok=True)
        self._saved_first_model_json = False
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_index = epoch + 1
        if epoch_index % self.period != 0:
            return
        weights_filename = self.weights_name_template.format(epoch=epoch_index)
        weights_path = os.path.join(self.dirpath, weights_filename)
        try:
            self.model.save_weights(weights_path)
            if self.verbose:
                print(f"[PeriodicSaveConfig] Saved weights to {weights_path}")
        except Exception as e:
            if self.verbose:
                print(f"[PeriodicSaveConfig] Failed to save weights: {e}")
        should_save_model_json = False
        if not self._saved_first_model_json and self.save_model_json_every is None:
            should_save_model_json = True
        elif isinstance(self.save_model_json_every, int) and (epoch_index % self.save_model_json_every == 0):
            should_save_model_json = True
        if should_save_model_json:
            try:
                model_json = self.model.to_json()
                model_json_path = os.path.join(self.dirpath, self.model_json_name)
                with open(model_json_path, 'w') as f:
                    f.write(model_json)
                self._saved_first_model_json = True
                if self.verbose:
                    print(f"[PeriodicSaveConfig] Saved model JSON to {model_json_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[PeriodicSaveConfig] Failed to save model JSON: {e}")
        info = {}
        info['epoch'] = int(epoch_index)
        info['logs'] = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v) for k,v in logs.items()}
        try:
            opt = self.model.optimizer
            try:
                info['optimizer_config'] = opt.get_config()
            except Exception:
                info['optimizer_config'] = str(opt)
            try:
                lr = opt.learning_rate
                if hasattr(lr, 'numpy'):
                    info['learning_rate'] = float(lr.numpy())
                else:
                    info['learning_rate'] = str(lr)
            except Exception:
                info['learning_rate'] = None
        except Exception:
            info['optimizer_config'] = None
            info['learning_rate'] = None
        training_info_filename = self.training_info_template.format(epoch=epoch_index)
        training_info_path = os.path.join(self.dirpath, training_info_filename)
        try:
            with open(training_info_path, 'w') as f:
                json.dump(info, f, indent=2)
            if self.verbose:
                print(f"[PeriodicSaveConfig] Saved training info to {training_info_path}")
        except Exception as e:
            if self.verbose:
                print(f"[PeriodicSaveConfig] Failed to save training info: {e}")
