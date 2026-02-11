import tensorflow as tf

class StepLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, step_size, gamma, steps_per_epoch):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch
    def __call__(self, step):
        exponent = tf.floor(tf.cast(step, tf.float32) / tf.cast(self.step_size, tf.float32))
        return self.initial_lr * tf.pow(self.gamma, exponent)
