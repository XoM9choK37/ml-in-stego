import numpy as np
import tensorflow as tf
import os
import json

class PeriodicFullSaveConfig(tf.keras.callbacks.Callback):
    def __init__(self, dirpath='model_checkpoints', period=1,
                 model_name_template='model_epoch_{epoch:04d}',
                 training_info_template='training_info_epoch_{epoch:04d}.json',
                 save_format='tf',  # 'tf' для SavedModel, 'h5' для HDF5
                 save_weights_also=False,  # дополнительно сохранять только веса
                 verbose=1):
        """
        Callback для периодического сохранения ПОЛНОЙ модели.
        
        Args:
            dirpath: директория для сохранения
            period: каждые сколько эпох сохранять
            model_name_template: шаблон имени для полной модели
            training_info_template: шаблон для файлов с информацией
            save_format: формат сохранения ('tf' для SavedModel, 'h5' для HDF5)
            save_weights_also: дополнительно сохранять только веса (для совместимости)
            verbose: подробный вывод
        """
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        
        self.dirpath = dirpath
        self.period = int(period)
        self.model_name_template = model_name_template
        self.training_info_template = training_info_template
        self.save_format = save_format
        self.save_weights_also = save_weights_also
        self.verbose = verbose
        
        # Создаем директорию
        os.makedirs(self.dirpath, exist_ok=True)
        
    def _convert_to_serializable(self, obj):
        """Конвертирует numpy типы в стандартные Python типы для JSON"""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (tf.Tensor, tf.Variable)):
            return obj.numpy().tolist() if hasattr(obj, 'numpy') else str(obj)
        return obj
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_index = epoch + 1
        
        # Проверяем, нужно ли сохранять на этой эпохе
        if epoch_index % self.period != 0:
            return
        
        # 1. Сохраняем ПОЛНУЮ модель (со всеми статистиками BN)
        model_filename = self.model_name_template.format(epoch=epoch_index)
        model_path = os.path.join(self.dirpath, model_filename)
        
        try:
            # Добавляем расширение в зависимости от формата
            if self.save_format == 'h5':
                save_path = model_path + '.h5'
            else:
                save_path = model_path  # для SavedModel создается папка
            
            # Сохраняем полную модель
            self.model.save(save_path, save_format=self.save_format)
            
            if self.verbose:
                print(f"\n[PeriodicFullSaveConfig] Полная модель сохранена в {save_path}")
                
            # 2. Опционально: сохраняем только веса для обратной совместимости
            if self.save_weights_also:
                weights_path = os.path.join(
                    self.dirpath, 
                    f'weights_only_epoch_{epoch_index:04d}.h5'
                )
                self.model.save_weights(weights_path)
                if self.verbose:
                    print(f"[PeriodicFullSaveConfig] Веса также сохранены в {weights_path}")
                    
        except Exception as e:
            if self.verbose:
                print(f"[PeriodicFullSaveConfig] Ошибка при сохранении модели: {e}")
        
        # 3. Сохраняем информацию о тренировке (метрики, learning rate и т.д.)
        info = {
            'epoch': int(epoch_index),
            'logs': self._convert_to_serializable(logs),
            'model_path': model_path,
            'save_format': self.save_format
        }
        
        # Сохраняем информацию об оптимизаторе
        try:
            opt = self.model.optimizer
            if opt is not None:
                # Конфигурация оптимизатора
                try:
                    opt_config = opt.get_config()
                    info['optimizer_config'] = self._convert_to_serializable(opt_config)
                except Exception:
                    info['optimizer_config'] = str(opt)
                
                # Текущий learning rate
                try:
                    lr = opt.learning_rate
                    if hasattr(lr, 'numpy'):
                        info['learning_rate'] = float(lr.numpy())
                    elif isinstance(lr, (tf.Variable, tf.Tensor)):
                        info['learning_rate'] = float(lr.numpy())
                    else:
                        info['learning_rate'] = float(lr)
                except Exception:
                    info['learning_rate'] = None
                    
                # Количество шагов
                if hasattr(opt, 'iterations'):
                    info['iterations'] = int(opt.iterations.numpy())
        except Exception as e:
            info['optimizer_info'] = f"Error getting optimizer info: {e}"
        
        # Сохраняем информацию о модели
        try:
            # Считаем количество параметров
            trainable_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
            non_trainable_params = np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights])
            
            info['model_summary'] = {
                'trainable_params': int(trainable_params),
                'non_trainable_params': int(non_trainable_params),
                'total_params': int(trainable_params + non_trainable_params),
                'layers': len(self.model.layers)
            }
            
            # Информация о Batch Normalization слоях
            bn_info = []
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    bn_info.append({
                        'layer_name': layer.name,
                        'index': i,
                        'center': layer.center,
                        'scale': layer.scale,
                        'momentum': layer.momentum,
                        'has_moving_mean': hasattr(layer, 'moving_mean'),
                        'has_moving_variance': hasattr(layer, 'moving_variance')
                    })
            if bn_info:
                info['batch_norm_layers'] = bn_info
                
        except Exception as e:
            info['model_info_error'] = str(e)
        
        # Сохраняем JSON с информацией
        training_info_filename = self.training_info_template.format(epoch=epoch_index)
        training_info_path = os.path.join(self.dirpath, training_info_filename)
        
        try:
            with open(training_info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, default=self._convert_to_serializable)
            if self.verbose:
                print(f"[PeriodicFullSaveConfig] Информация сохранена в {training_info_path}")
        except Exception as e:
            if self.verbose:
                print(f"[PeriodicFullSaveConfig] Ошибка при сохранении информации: {e}")

    def on_train_end(self, logs=None):
        """Сохраняем финальную модель после окончания обучения"""
        if self.verbose:
            print("\n[PeriodicFullSaveConfig] Обучение завершено, сохраняем финальную модель...")
        
        # Сохраняем финальную модель
        final_model_path = os.path.join(self.dirpath, 'final_model')
        try:
            self.model.save(final_model_path, save_format=self.save_format)
            if self.verbose:
                print(f"[PeriodicFullSaveConfig] Финальная модель сохранена в {final_model_path}")
        except Exception as e:
            if self.verbose:
                print(f"[PeriodicFullSaveConfig] Ошибка при сохранении финальной модели: {e}")
                