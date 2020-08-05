import torch.nn as nn


class SimpleGenerator(nn.Module):
    def __init__(self, n_z, hiddens, output_dim):
        super().__init__()
        total_out = 1
        for dim in output_dim:
            total_out *= dim

        self.output_dim = output_dim
        self.hiddens = hiddens
        self.n_z = n_z

        self.main = nn.Sequential()
        for i, (n_in, n_out) in enumerate(zip([n_z] + hiddens, hiddens + [total_out])):
            self.main.add_module(f'FC_{i}', nn.Linear(n_in, n_out))
            if i < len(hiddens):
                self.main.add_module(f'ReLU_{i}', nn.ReLU(inplace=True))
            else:
                self.main.add_module(f'Tanh_{i}', nn.Tanh())

    def forward(self, input):
        out = self.main(input)
        return out.reshape(-1, *self.output_dim)


class SimpleDiscriminator(nn.Module):
    def __init__(self, input_dim, hiddens):
        super().__init__()
        self.input_dim = input_dim
        self.hiddens = hiddens

        self.main = nn.Sequential()
        for i, (n_in, n_out) in enumerate(zip([input_dim] + hiddens, hiddens + [1])):
            self.main.add_module(f'FC_{i}', nn.Linear(n_in, n_out))
            if i < len(hiddens):
                self.main.add_module(f'LeakyReLU_{i}', nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        out = input.view(-1, self.input_dim)
        return self.main(out)


class GeneratorMNIST(nn.Module):
    """
        Selected Generator for MNIST dataset.
        This produces images of size 28 x 28.
        Use this module for any GANs to be trained on MNIST.
    """
    def __init__(self, n_z):
        """
        Function to construct a Generator instance
        Args:
            n_z : Dimensionality of the noise
        """
        super(GeneratorMNIST, self).__init__()

        assert n_z > 0, "Dimensionality of the noise vector has to be positive"

        # Architecture: Specified as follows:
        # |   INPUT     ---->      OUTPUT    (         ACTIVATIONS          ) |
        # |    n_z      ---->       4096     (          LEAKY_RELU          ) |
        # |  4X4X256    ---->     7X7X128    (       BATCHNORM_2D, RELU     ) |
        # |  7x7x128    ---->     14x14x64   (       BATCHNORM_2D, RELU     ) |
        # |  14X14X64   ---->     28X28X1    (       BATCHNORM_2D, TANH     ) |
        leaky_coeff = 0.2

        # Fully connected Section
        layer = 1
        main = nn.Sequential()
        main.add_module('Linear_{0}-{1}-{2}'.format(layer, n_z, 4096),
                        nn.Linear(n_z, 4096))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))
        self.fc = main

        # Convolution Section
        layer = layer + 1
        main = nn.Sequential()
        main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 256, 128),
                        nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                           kernel_size=4, stride=1, padding=0))
        main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        layer = layer + 1
        main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 128, 64),
                        nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        layer = layer + 1
        main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 64, 1),
                        nn.ConvTranspose2d(in_channels=64, out_channels=1,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module('TanH_{0}'.format(layer), nn.Tanh())
        self.cv = main

        self.n_z = n_z

    def forward(self, input):
        input = input.view(-1, self.n_z)
        pass_ = self.fc(input)
        pass_ = pass_.view(-1, 256, 4, 4)
        pass_ = self.cv(pass_)
        return pass_


class DiscriminatorMNIST(nn.Module):
    """
        Selected Discriminator for MNIST dataset. This discriminates images of
        size 28 x 28.
        Use this module for any GANs to be trained on MNIST.
    """
    def __init__(self):
        """
        Function to construct a Discriminator instance
        """
        super(DiscriminatorMNIST, self).__init__()

        # Architecture: Specified as follows:
        # |   INPUT     ---->      OUTPUT    (           ACTIVATIONS        ) |
        # |  28x28x1    ---->     14x14x64   (   BATCHNORM_2D, LEAKY_RELU   ) |
        # |  14X14X64   ---->     7X7X128    (   BATCHNORM_2D, LEAKY_RELU   ) |
        # |  7x7x128    ---->     4x4x256    (   BATCHNORM_2D, LEAKY_RELU   ) |
        # |   4096      ---->        1       (            SIGMOID           ) |
        leaky_coeff = 0.2

        # Convolution Section
        layer = 1
        main = nn.Sequential()
        main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 1, 64),
                        nn.Conv2d(in_channels=1, out_channels=64,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 64),
                        nn.BatchNorm2d(64))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))

        layer = layer + 1
        main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 64, 128),
                        nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 128),
                        nn.BatchNorm2d(128))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))

        layer = layer + 1
        main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 128, 256),
                        nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=4, stride=1, padding=0))
        main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 256),
                        nn.BatchNorm2d(256))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))
        self.cv = main

        # Fully connected Section
        layer = layer + 1
        main = nn.Sequential()
        main.add_module('Linear_{0}-{1}-{2}'.format(layer, 4096, 1),
                        nn.Linear(4096, 1))
        self.fc = main

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        pass_ = self.cv(input)
        pass_ = pass_.view(-1, 4096)
        pass_ = self.fc(pass_)
        return pass_


class GeneratorCIFAR10(nn.Module):
    """
        Selected Generator for CIFAR10 dataset.
        This produces images of size 32 x 32.
        Use this module for any GANs to be trained on CIFAR10.
    """
    def __init__(self, n_z):
        """
        Function to construct a Generator instance
        Args:
            n_z  : Dimensionality of the noise
        """
        super(GeneratorCIFAR10, self).__init__()
        assert n_z > 0, "Dimensionality of the noise vector has to be positive"

        # Architecture: Specified as follows:
        # |   INPUT     ---->      OUTPUT    (         ACTIVATIONS         ) |
        # |    n_z      ---->       4096     (         LEAKY_RELU          ) |
        # |  4x4x256    ---->     8X8X128    (       BATCHNORM_2D, RELU    ) |
        # |  8x8x128    ---->     16x16x64   (       BATCHNORM_2D, RELU    ) |
        # |  16X16X64   ---->     32X32X3    (       BATCHNORM_2D, TANH    ) |
        leaky_coeff = 0.2

        # Fully Connected Section
        layer = 1
        main = nn.Sequential()
        main.add_module('Linear_{0}-{1}-{2}'.format(layer, n_z, 4096),
                        nn.Linear(n_z, 4096))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))
        self.fc = main

        # Convolutional Section
        layer = layer + 1
        main = nn.Sequential()
        main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 256, 128),
                        nn.ConvTranspose2d(in_channels=256,
                                           out_channels=128,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))
        layer = layer + 1
        main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 128, 64),
                        nn.ConvTranspose2d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module('ReLU_{0}'.format(layer), nn.ReLU(True))

        layer = layer + 1
        main.add_module('ConvTranspose2d_{0}-{1}-{2}'.format(layer, 64, 3),
                        nn.ConvTranspose2d(in_channels=64,
                                           out_channels=3,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module('TanH_{0}'.format(layer), nn.Tanh())
        self.cv = main

        self.n_z = n_z

    def forward(self, input):
        input = input.view(-1, self.n_z)
        pass_ = self.fc(input)
        pass_ = pass_.view(-1, 256, 4, 4)
        pass_ = self.cv(pass_)
        return pass_


class DiscriminatorCIFAR10(nn.Module):
    """
        Selected Discriminator for CIFAR10 dataset.
        This discriminates images of size 32 x 32.
        Use this module for any GANs to be trained on CIFAR10.
    """
    def __init__(self):
        """
        Function to construct a Discriminator instance
        Args:
            ngpu : Number of GPUs to be used
        """
        super(DiscriminatorCIFAR10, self).__init__()

        # Architecture: Specified as follows:
        # |   INPUT     ---->      OUTPUT   (          ACTIVATIONS         ) |
        # |  32x32x3    ---->     16x16x64  (    BATCHNORM_2D, LEAKY_RELU  ) |
        # |  16X16X64   ---->     8X8X128   (    BATCHNORM_2D, LEAKY_RELU  ) |
        # |  8x8x128    ---->     4x4x256   (    BATCHNORM_2D, LEAKY_RELU  ) |
        # |   4096      ---->      1        (            SIGMOID           ) |
        leaky_coeff = 0.2
        # Convolution Section
        layer = 1
        main = nn.Sequential()
        main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 3, 64),
                        nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 64),
                        nn.BatchNorm2d(64))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))

        layer = layer + 1
        main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 64, 128),
                        nn.Conv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 128),
                        nn.BatchNorm2d(128))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))

        layer = layer + 1
        main.add_module('Conv2d_{0}-{1}-{2}'.format(layer, 128, 256),
                        nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module('BatchNorm2d_{0}-{1}'.format(layer, 256),
                        nn.BatchNorm2d(256))
        main.add_module('LeakyReLU_{0}'.format(layer),
                        nn.LeakyReLU(leaky_coeff, inplace=True))
        self.cv = main

        # Fully connected Section
        layer = layer + 1
        main = nn.Sequential()
        main.add_module('Linear_{0}-{1}-{2}'.format(layer, 4096, 1),
                        nn.Linear(4096, 1))
        self.fc = main

    def forward(self, input):
        input = input.view(-1, 3, 32, 32)
        pass_ = self.cv(input)
        pass_ = pass_.view(-1, 4096)
        pass_ = self.fc(pass_)
        return pass_


class GeneratorDCGANCIFAR10(nn.Module):
    def __init__(self, n_z):
        """
        Function to construct a Generator instance
        Args:
            n_z  : Dimensionality of the noise
        """
        super(GeneratorDCGANCIFAR10, self).__init__()
        assert n_z > 0, "Dimensionality of the noise vector has to be positive"
        self.n_z = n_z

        main = nn.Sequential()
        layer = 1
        main.add_module(f'Linear_{layer}_{n_z}_{4096}', nn.Linear(self.n_z, 4096))
        main.add_module(f'BatchNorm1d_{layer}_{4096}', nn.BatchNorm1d(4096))
        main.add_module(f'ReLU_{layer}', nn.ReLU(inplace=True))
        self.fc = main

        main = nn.Sequential()
        layer += 1
        main.add_module(f'ConvTranspose2d_{layer}-{256}-{128}',
                        nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module(f'BatchNorm2d_{layer}_{128}', nn.BatchNorm2d(128))
        main.add_module(f'ReLU_{layer}', nn.ReLU(inplace=True))

        layer += 1
        main.add_module(f'ConvTranspose2d_{layer}-{128}-{64}',
                        nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module(f'BatchNorm2d_{layer}_{64}', nn.BatchNorm2d(64))
        main.add_module(f'ReLU_{layer}', nn.ReLU(inplace=True))

        layer += 1
        main.add_module('ConvTranspose2d_{layer}-{64}-{3}',
                        nn.ConvTranspose2d(in_channels=64, out_channels=3,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module('TanH_{0}'.format(layer), nn.Tanh())
        self.conv = main

    def forward(self, input):
        input = input.view(-1, self.n_z)
        pass_ = self.fc(input)
        pass_ = pass_.view(-1, 256, 4, 4)
        pass_ = self.conv(pass_)
        return pass_


class DiscriminatorDCGANCIFAR10(nn.Module):
    def __init__(self):
        """
        Function to construct a Discriminator instance
        """
        super(DiscriminatorDCGANCIFAR10, self).__init__()

        main = nn.Sequential()
        layer = 1
        main.add_module(f'Conv2d_{layer}_{3}_{64}',
                        nn.Conv2d(in_channels=3, out_channels=64,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module(f'LeakyReLU_{layer}', nn.LeakyReLU(0.2, inplace=True))

        layer += 1
        main.add_module(f'Conv2d_{layer}_{64}_{128}',
                        nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module(f'BatchNorm2d_{layer}_{128}', nn.BatchNorm2d(128))
        main.add_module(f'LeakyReLU_{layer}', nn.LeakyReLU(0.2, inplace=True))

        layer += 1
        main.add_module(f'Conv2d_{layer}_{128}_{256}',
                        nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module(f'BatchNorm2d_{layer}_{256}', nn.BatchNorm2d(256))
        main.add_module(f'LeakyReLU_{layer}', nn.LeakyReLU(0.2, inplace=True))
        self.conv = main

        layer = layer + 1
        main = nn.Sequential()
        main.add_module('Linear_{0}-{1}-{2}'.format(layer, 4096, 1),
                        nn.Linear(4096, 1))
        self.fc = main

    def forward(self, input):
        input = input.view(-1, 3, 32, 32)
        pass_ = self.conv(input)
        pass_ = pass_.view(-1, 4096)
        pass_ = self.fc(pass_)
        return pass_


class GeneratorDCGANMNIST(nn.Module):
    def __init__(self, n_z):
        """
        Function to construct a Generator instance
        Args:
            n_z  : Dimensionality of the noise
        """
        super(GeneratorDCGANMNIST, self).__init__()
        assert n_z > 0, "Dimensionality of the noise vector has to be positive"
        self.n_z = n_z

        main = nn.Sequential()
        layer = 1
        main.add_module(f'Linear_{layer}_{n_z}_{4096}', nn.Linear(self.n_z, 4096))
        main.add_module(f'BatchNorm1d_{layer}_{4096}', nn.BatchNorm1d(4096))
        main.add_module(f'ReLU_{layer}', nn.ReLU(inplace=True))
        self.fc = main

        main = nn.Sequential()
        layer += 1
        main.add_module(f'ConvTranspose2d_{layer}-{256}-{128}',
                        nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                           kernel_size=4, stride=1, padding=0))
        main.add_module(f'BatchNorm2d_{layer}_{128}', nn.BatchNorm2d(128))
        main.add_module(f'ReLU_{layer}', nn.ReLU(inplace=True))

        layer += 1
        main.add_module(f'ConvTranspose2d_{layer}-{128}-{64}',
                        nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module(f'BatchNorm2d_{layer}_{64}', nn.BatchNorm2d(64))
        main.add_module(f'ReLU_{layer}', nn.ReLU(inplace=True))

        layer += 1
        main.add_module('ConvTranspose2d_{layer}-{64}-{3}',
                        nn.ConvTranspose2d(in_channels=64, out_channels=1,
                                           kernel_size=4, stride=2, padding=1))
        main.add_module('TanH_{0}'.format(layer), nn.Tanh())
        self.conv = main

    def forward(self, input):
        input = input.view(-1, self.n_z)
        pass_ = self.fc(input)
        pass_ = pass_.view(-1, 256, 4, 4)
        pass_ = self.conv(pass_)
        return pass_


class DiscriminatorDCGANMNIST(nn.Module):
    def __init__(self):
        """
        Function to construct a Discriminator instance
        """
        super(DiscriminatorDCGANMNIST, self).__init__()

        main = nn.Sequential()
        layer = 1
        main.add_module(f'Conv2d_{layer}_{1}_{64}',
                        nn.Conv2d(in_channels=1, out_channels=64,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module(f'LeakyReLU_{layer}', nn.LeakyReLU(0.2, inplace=True))

        layer += 1
        main.add_module(f'Conv2d_{layer}_{64}_{128}',
                        nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=4, stride=2, padding=1))
        main.add_module(f'BatchNorm2d_{layer}_{128}', nn.BatchNorm2d(128))
        main.add_module(f'LeakyReLU_{layer}', nn.LeakyReLU(0.2, inplace=True))

        layer += 1
        main.add_module(f'Conv2d_{layer}_{128}_{256}',
                        nn.Conv2d(in_channels=128, out_channels=256,
                                  kernel_size=4, stride=1, padding=0))
        main.add_module(f'BatchNorm2d_{layer}_{256}', nn.BatchNorm2d(256))
        main.add_module(f'LeakyReLU_{layer}', nn.LeakyReLU(0.2, inplace=True))
        self.conv = main

        layer = layer + 1
        main = nn.Sequential()
        main.add_module('Linear_{0}-{1}-{2}'.format(layer, 4096, 1),
                        nn.Linear(4096, 1))
        self.fc = main

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        pass_ = self.conv(input)
        pass_ = pass_.view(-1, 4096)
        pass_ = self.fc(pass_)
        return pass_
