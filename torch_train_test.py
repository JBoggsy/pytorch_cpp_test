from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import malmoenv

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as tf

IMAGE_CHANS = 4
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ngpu = 1
        self.layer_1 = nn.Sequential(
            nn.Conv2d(IMAGE_CHANS, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.layer_2_resid = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer_3_resid = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer_4_resid = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer_5_resid = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 2, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.layer_6 = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, 2, 1, bias=False),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2048, 2048, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.layer_6_resid = nn.Sequential(
            nn.Conv2d(1024, 2048, 1, 2, bias=False),
            nn.BatchNorm2d(2048)
        )

        self.flatten_layer = nn.Flatten()

    def forward(self, input):
        r1 = self.layer_1(input)

        # r2_prime = self.layer_2(r1)
        # r2_resid = self.layer_2_resid(r1)
        # r2 = r2_prime + r2_resid

        # r3_prime = self.layer_3(r2)
        # r3_resid = self.layer_3_resid(r2)
        # r3 = r3_prime + r3_resid

        # r4_prime = self.layer_4(r3)
        # r4_resid = self.layer_4_resid(r3)
        # r4 = r4_prime + r4_resid

        # r5_prime = self.layer_5(r4)
        # r5_resid = self.layer_5_resid(r4)
        # r5 = r5_prime + r5_resid

        # r6_prime = self.layer_6(r5)
        # r6_resid = self.layer_6_resid(r5)
        # r6 = r6_prime + r6_resid

        out = self.flatten_layer(r1)
        return out

class EncoderCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.main = nn.Sequential(
            # nn.Linear(131072, 65536, bias=False),
            # nn.ReLU(),
            # nn.Linear(65536, 32768, bias=False),
            # nn.ReLU(),
            # nn.Linear(32768, 32768, bias=False),
            # nn.ReLU(),
            # nn.Linear(32768, 65536, bias=False),
            # nn.ReLU(),
            # nn.Linear(65536, 131072, bias=False)
            nn.Linear(131072, 131072, bias=False),
            nn.ReLU()
        )

    def forward(self, input):
        return self.main(input)

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ngpu = 1
        self.unflatten_layer = nn.Unflatten(1, (64, 256, 256))

        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.layer_1_resid = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024)
        )
        
        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer_2_resid = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.layer_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer_3_resid = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256)
        )
        
        self.layer_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.layer_4_resid = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.layer_5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer_5_resid = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.layer_6 = nn.Sequential(
            nn.ConvTranspose2d(64, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
        )
        self.layer_6_resid = nn.Sequential(
            nn.ConvTranspose2d(64, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4)
        )

        self.force_rgb_range_layer = nn.Sigmoid()

    def forward(self, input):
        r0 = self.unflatten_layer(input)
        
        # r1_prime = self.layer_1(r0)
        # r1_resid = self.layer_1_resid(r0)
        # r1 = r1_prime + r1_resid
        
        # r2_prime = self.layer_2(r1)
        # r2_resid = self.layer_2_resid(r1)
        # r2 = r2_prime + r2_resid
        
        # r3_prime = self.layer_3(r2)
        # r3_resid = self.layer_3_resid(r2)
        # r3 = r3_prime + r3_resid
        
        # r4_prime = self.layer_4(r3)
        # r4_resid = self.layer_4_resid(r3)
        # r4 = r4_prime + r4_resid
        
        # r5_prime = self.layer_5(r4)
        # r5_resid = self.layer_5_resid(r4)
        # r5 = r5_prime + r5_resid
        
        r6_prime = self.layer_6(r0)
        r6_resid = self.layer_6_resid(r0)
        r6 = r6_prime + r6_resid

        out = self.force_rgb_range_layer(r6)
        return out

if __name__ == "__main__":
    mission_file = Path("random_world.xml")
    env = malmoenv.make()
    # env.init(xml=mission_file.read_text().replace("FORCE_RESET", "true"),
    #         server=None,
    #         port=9000,
    #         action_filter={})
    # obs = env.reset()
    # obs = obs.reshape(512, 512,4)
    # obs = np.flip(obs, axis=0)
    # ax = plt.subplot()
    # ax.imshow(obs)
    # plt.show()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")
    encoder = Encoder().to(device)
    encoder.apply(weights_init)

    # core_encoder = EncoderCore().to(device)
    # core_encoder.apply(weights_init)

    decoder = Decoder().to(device)
    decoder.apply(weights_init)

    # obs_tensor = tf.to_tensor(obs.copy()).reshape((1, 4, 512, 512)).to(device)
    # print(obs_tensor.shape)

    # encoding = encoder(obs_tensor)
    # core_encoding = core_encoder(encoding)
    # decoding = decoder(encoding)

    # decoded_img = decoding.cpu().detach().numpy()
    # decoded_img = np.transpose(np.squeeze(decoded_img), (1,2,0))
    # ax = plt.subplot()
    # ax.imshow(decoded_img)
    # plt.show()

    criterion = nn.CrossEntropyLoss()
    optimizerE = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for i in range(1):
        # env.init(xml=mission_file.read_text().replace("FORCE_RESET", "true"),
        # server=None,
        # port=9000,
        # action_filter={})
        # env.reset()
        env.init(xml=mission_file.read_text().replace("FORCE_RESET", "false"),
        server=None,
        port=9000,
        action_filter={})
        env.reset()
        print(list(env.action_space))
        for j in range(100):
            action = env.action_space.sample()
            print(action)
            obs = obs, reward, done, info = env.step(action)
            if done:
                break
            obs = obs.reshape(512, 512,4)
            obs = np.flip(obs, axis=0)
            obs_tensor = tf.to_tensor(obs.copy()).reshape((1, 4, 512, 512)).to(device)
            encoding = encoder(obs_tensor)
            decoding = decoder(encoding)
            loss = criterion(decoding, obs_tensor)
            loss.backward()
            optimizerE.step()
            optimizerD.step()

            print(f"{i}.{j} loss = {loss}")
            decoded_img = decoding.cpu().detach().numpy()
            decoded_img = np.transpose(np.squeeze(decoded_img), (1,2,0))
            cv2.imshow("Input", obs)
            cv2.imshow("Output", decoded_img)
            cv2.waitKey(1)
        # decoded_img = decoding.cpu().detach().numpy()
        # decoded_img = np.transpose(np.squeeze(decoded_img), (1,2,0))
        # ax = plt.subplot()
        # ax.imshow(decoded_img)
        # plt.show()