# Binomial Mixture Model with Expectation-Maximization Algorithm

I write an article on this model on medium.com. This is [the link](https://medium.com/@jingluan.xw/binomial-mixture-model-with-expectation-maximum-em-algorithm-feeaf0598b60). If you are interested in the derivations of the equations used in this code and the article, please read [my note](https://www.dropbox.com/s/fy2kq9eanhwinpr/Binomial_Mixture_Model_EL_Algorithm_Derivations%20%281%29.pdf?dl=0).

In the notebook `BMM_EM_Algorithm.ipynb`, I first generate random data according to two mixed binomial distribution. Then I use the EL Algorithm to fit a BMM for the data. This is a demonstration on how to implement the EL Algorithm to build up a BMM model.

In the notebook `BMM_EM_Algorithm_fit_K_torch.ipynb`, I take use of torch, which is very good at matrix operations. The matrix operations naturally replace loops, meaning that a loop now can be executed in a parallel way. Thus torch speeds up the calculation significantly. Another advantage of torch is that it allows the usage of gpu, which further speeds up the calculation.

The second notebook, `BMM_EM_Algorithm_fit_K_torch.ipynb`, also deals with the choice for K, the number of components in a BMM. Say, K=3, meaning that there are three different binomial distributions in the mixture model. The optimal `K` is chosen based on Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC). The AIC and BIC decreases as `K` increases, and they flatten (or stops decreasing dramatically) with increasing `K` at the some value. Then this value is chosen as the optimal `K`.

# Author
Jing Luan -- jingluan.xw at gmail dot com

# Citing this code
Please cite this repository, when using this code.

# Licensing

Copyright 2019 by Jing Luan.

In brief, you can use, distribute, and change this package as you please.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Acknowledgement

Jing Luan is supported by Association of Members of the Institute for Advanced Study (Amias) while developing this project.
