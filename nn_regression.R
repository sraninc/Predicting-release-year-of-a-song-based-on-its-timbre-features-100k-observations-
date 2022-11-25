library(keras)
library (dplyr)
library(tensorflow)

FLAGS <- flags(
  flag_numeric('dropout1', 0.3),
  flag_numeric('dropout2', 0.3),
  flag_integer('nodes1', 128),
  flag_integer('nodes2', 128),
  flag_integer('nodes3', 128),
  flag_numeric('l2', 0.001),
  flag_string ('optimizer', 'rmsprop'),
  flag_numeric('lr', 0.1),
  flag_numeric ('batch_size', 100),
  flag_numeric ('epochs', 200)
)

early_stop = callback_early_stopping(monitor = "val_loss", patience = 5)

model = keras_model_sequential()
model %>% 
  layer_dense(units = FLAGS$nodes1, 
              input_shape = ncol(x),
              activation = 'relu',
              kernel_regularizer = regularizer_l2(l = FLAGS$l2)) %>%
  layer_dropout(FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$nodes2,
              activation = 'relu') %>%
  layer_dropout(FLAGS$dropout2) %>%
  layer_dense(units = FLAGS$nodes3,
              activation = 'relu') %>%
  layer_dense(units = 1) %>%
  
  
  
  compile (optimizer = FLAGS$optimizer,
           loss = "mean_squared_error",
           metrics = c("mean_squared_error") )  %>%
  fit (x, y, 
       epochs = FLAGS$epochs, 
       batch_size = FLAGS$batch_size,
       validation_split = 0.2,
       verbose = 0,
       callbacks = list(early_stop, 
                        callback_reduce_lr_on_plateau(factor = FLAGS$lr)))
