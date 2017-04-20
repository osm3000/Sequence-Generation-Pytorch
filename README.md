# Time sequence generation using PyTorch
This is an attempt to familiarize myself with PyTorch. In this example, the target to generate a sequence of continuous data (sine waves or mix of them) using LSTM

## Updates
* 16/04/2017: When trying to generate a simple sine wave, the system flats
out. It is unclear for me why this happens. The same happens with 2 and 3 sine-wave 
components.
* 18/04/2017: Thanks to the advice of  Sean Robertson - https://discuss.pytorch.org/t/lstm-time-sequence-generation/1916/4 - to reduce the 
frequency of the sine-waves, I was finally able to generate signals. The 2 and 3
sine-wave components are working well (the 3 is a bit unstable).
* 19/04/2017: The method of teaching the model using only the ground truth is called 
 **Teacher Forcing**.
* 20/04/2017: After further testing , I found my model works when the sine 
wave has a relatively high frequency (1/60 Hz or more). Lower frequency like 
(1/180 Hz) doesn't work.
    * With a sequence length of 100 timesteps, the model 
flats out when I use it for generation. 
    * I tried to increase the sequence length till 500. The model no longer flats 
    out, but the performance is poor. Probably the dependency is too long for the 
    model to remember.
        * I need a way to be definite about this issue
## TODO:
- [x] Train the model on generation instead of prediction: training the model
on its own output
    * Strangely, it doesn't lead to different results. With low frequencies, it doesn't work. With higher frequency, its
    performance is almost similar to the naive approach (where I train on the ground)
    truth.
- [ ] Try Bengio approach DAD `Scheduled Sampling`
    * Not optimistic though

