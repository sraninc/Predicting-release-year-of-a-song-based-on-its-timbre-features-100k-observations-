library (dplyr)

train_flags <- flags(
  flag_numeric('dropout1', 0.3),
  flag_integer('nodes1', 128),
  flag_integer('nodes2', 128),
  flag_integer('nodes3', 128),
  flag_numeric('l2', 0.001),
  flag_string ('optimizer', 'rmsprop'),
  flag_numeric('lr', 0.1)
)

early_stop = callback_early_stopping(monitor = "val_loss", patience = 5)

model <- keras_model_sequential() %>%
  layer_dense(units = train_flags$nodes1, input_shape = ncol(x), activation = 'relu', kernel_regularizer = regularizer_l2(l = train_flags$l2)) %>%
  layer_dense(units = train_flags$nodes2,
              activation = 'relu') %>%
  layer_dropout(train_flags$dropout1) %>%
  layer_dense(units = train_flags$nodes3,
              activation = 'relu') %>%
  layer_dense(units = 3, activation = "softmax") %>%

  
 

  compile (optimizer = optimizer_rmsprop(),
                   loss = "categorical_crossentropy",
                   metrics = c("accuracy") ) %>%
  fit (x, train_labels, 
       epochs = 200, 
       batch_size = 1000,
       validation_split = 0.2,
       verbose = 0,
       callbacks = list(early_stop, callback_reduce_lr_on_plateau(factor = train_flags$lr)))

    
    
  
  
  
