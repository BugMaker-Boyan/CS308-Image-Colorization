# CS308-Image-Colorization
Image colorization assigns a color to each pixel of a target grayscale image. It is a classical
problem in computer visual, which need to use a grayscale photograph as input, and expected
output a a plausible color version of the photograph. This problem is somewhat challenging
because it is multimodal -- a single grayscale image may correspond to many reasonable color
images.

Generative adversarial networks (GANs) is a type of generative model. A GAN is composed of two
smaller networks called the generator and discriminator. The generator’s task is to produce
results that are indistinguishable from real data. The discriminator’s task is to classify whether a
sample came from the generator’s model distribution or the original data distribution. Both of
these subnetworks are trained simultaneously until the generator is able to consistently produce
results that the discriminator cannot classify.
Conditional generative adversarial networks (conditional GAN) is a kind of GAN to address the
problem that the input of the generator is randomly generated noise data z in a traditional GAN,
which is not applicable to the automatic colorization problem because grayscale images serve as
the inputs of our problem rather than noise.
In our baseline model, we use U-Net to find a direct mapping from the grayscale image space to
color image space. The architecture of the model is symmetric, with n encoding units and n
decoding units. The contracting path consists of 4 × 4 convolution layers with stride 2 for
downsampling, each followed by batch normalization and Leaky-ReLU activation function with the
slope of 0.2. The number of channels are doubled after each step. Each unit in the expansive path
consists of a 4 × 4 transposed convolutional layer with stride 2 for upsampling, concatenation with
the activation map of the mirroring layer in the contracting path, followed by batch normalization
and ReLU activation function. The last layer of the network is a 1 × 1 convolution which is
equivalent to cross-channel parametric pooling layer. We use tanh function for the last layer. The
number of channels in the output layer is 3 with Lab* color space. We train the baseline model to
minimize the Euclidean distance between predicted and ground truth averaged over all pixels:
where x is our grayscale input image, y is the corresponding color image, p and l are indices of
pixels and color channels respectively, n is the total number of pixels, and h is a function mapping
from grayscale to color images.
We use Deep Convolutional GANs (DCGAN) for the generator and discriminator models and the
architecture was also modifified as a conditional GAN instead of a traditional DCGAN. The
architecture of generator G is the same as the baseline model. For discriminator D, we use similar
architecture as the baselines contractive path: a series of 4 × 4 convolutional layers with stride 2
with the number of channels being doubled after each downsampling. All convolution layers are
followed by batch normalization, leaky ReLU activation with slope 0.2. After the last layer, a
convolution is applied to map to a 1 dimensional output, followed by a sigmoid function to return
a probability value of the input being real or fake. The input of the discriminator is a colored
image either coming from the generator or true labels, concatenated with the grayscale image.
