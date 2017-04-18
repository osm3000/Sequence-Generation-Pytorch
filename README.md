# Time sequence generation using PyTorch
This is an attempt to familiarize myself with PyTorch. In this example, the target to generate a sequence of continuous data (sine waves or mix of them) using LSTM

## Updates
* 16/04/2017: When trying to generate a simple sine wave, the system flats
out. It is unclear for me why this happens. The same happens with 2 and 3 sine-wave 
components.
* 18/04/2017: Thanks to the advice of  Sean Robertson - https://discuss.pytorch.org/t/lstm-time-sequence-generation/1916/4 - to reduce the 
frequency of the sine-waves, I was finally able to generate signals. The 2 and 3
sine-wave components are working well (the 3 is a bit unstable).

## TODO:
* Train the model on generation instead of prediction: training the model
on its own output
