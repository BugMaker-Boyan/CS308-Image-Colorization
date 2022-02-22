import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder_conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False),
        )
        self.encoder_conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
        )
        self.encoder_conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
        )
        self.encoder_conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
        )
        self.encoder_conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
        )
        self.encoder_conv_6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
        )

        self.decoder_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
        )
        self.decoder_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
        )
        self.decoder_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.decoder_conv_4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.decoder_conv_5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )

        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, inputs):
        encoder_conv_1_outputs = self.encoder_conv_1(inputs)
        encoder_conv_2_outputs = self.encoder_conv_2(encoder_conv_1_outputs)
        encoder_conv_3_outputs = self.encoder_conv_3(encoder_conv_2_outputs)
        encoder_conv_4_outputs = self.encoder_conv_4(encoder_conv_3_outputs)
        encoder_conv_5_outputs = self.encoder_conv_5(encoder_conv_4_outputs)
        encoder_conv_6_outputs = self.encoder_conv_6(encoder_conv_5_outputs)

        decoder_conv_1_outputs = self.decoder_conv_1(encoder_conv_6_outputs)
        decoder_conv_2_outputs = self.decoder_conv_2(torch.cat([decoder_conv_1_outputs, encoder_conv_5_outputs], dim=1))
        decoder_conv_3_outputs = self.decoder_conv_3(torch.cat([decoder_conv_2_outputs, encoder_conv_4_outputs], dim=1))
        decoder_conv_4_outputs = self.decoder_conv_4(torch.cat([decoder_conv_3_outputs, encoder_conv_3_outputs], dim=1))
        decoder_conv_5_outputs = self.decoder_conv_5(torch.cat([decoder_conv_4_outputs, encoder_conv_2_outputs], dim=1))

        outputs = self.output(decoder_conv_5_outputs)

        return outputs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm2d(512),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.BatchNorm2d(512),
        )

        self.output = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.output(outputs)
        return outputs

