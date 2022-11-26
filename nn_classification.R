library (dplyr)

train_flags <- flags(
  flag_numeric('dropout1', 0.3),
  flag_numeric('dropout2', 0.3),
  flag_numeric('dropout3', 0.4),
  flag_integer('nodes1', 128),
  flag_integer('nodes2', 128),
  flag_integer('nodes3', 64),
  flag_numeric('l2', 0.001),
  flag_string ('optimizer', 'rmsprop'),
  flag_numeric('lr', 0.1),
  flag_numeric ('batch_size', 100),
  flag_numeric ('epochs', 200)
)

early_stop = callback_early_stopping(monitor = "val_loss", patience = 5)

model <- keras_model_sequential() %>%
  layer_dense(units = train_flags$nodes1, input_shape = ncol(x), activation = 'relu', kernel_regularizer = regularizer_l2(l = train_flags$l2)) %>%
  layer_dropout(train_flags$dropout1) %>%
  layer_dense(units = train_flags$nodes2,
              activation = 'relu') %>%
  layer_dropout(train_flags$dropout2) %>%
  layer_dense(units = train_flags$nodes3,
              activation = 'relu') %>%
  layer_dropout(train_flags$dropout3) %>%
  layer_dense(units = train_flags$nodes3,
              activation = 'relu') %>%
  layer_dense(units = 3, activation = "softmax") %>%

  
 

  compile (optimizer = optimizer_rmsprop(),
                   loss = "categorical_crossentropy",
                   metrics = c("accuracy") ) %>%
  fit (x, train_labels, 
       epochs = train_flags$epochs, 
       batch_size = train_flags$batch_size,
       validation_split = 0.2,
       verbose = 0,
       callbacks = list(early_stop, callback_reduce_lr_on_plateau(factor = train_flags$lr)))

    
    
  
  
  
