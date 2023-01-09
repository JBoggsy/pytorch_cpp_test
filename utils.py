import json
from pathlib import Path
import time

import torch
from torchvision.transforms import functional as tfunc
import torchvision.utils as vutils

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import MalmoPython

def draw_image(image, fname=None, show=False):
    if isinstance(image, torch.Tensor):
        if str(image.device) == 'cuda:0':
            image = image.detach().cpu()
        image = image.squeeze().numpy()
    plt.figure(figsize=(8,8))
    plt.axis("off")
    if len(image.shape) == 2:
        plt.imshow(image, interpolation="none")
    else:
        image = np.transpose(image, (1,2,0))
        plt.imshow(image, interpolation="none")
    if show:
        plt.show()

    if fname:
        plt.savefig(fname)
    plt.close()
    
def draw_layers(data, fname=None, show=False):
    if isinstance(data, torch.Tensor):
        if str(data.device) == 'cuda:0':
            data = data.detach().cpu()
        data = data.squeeze().numpy()
    if len(data.shape) == 2:
        return draw_image(data)
    data_layers = [tfunc.to_tensor(d) for d in data]
    grid_image = vutils.make_grid(data_layers, nrow=int(len(data_layers)**0.5), padding=0, pad_value=0.5, normalize=True).cpu()
    draw_image(grid_image, fname, show)


class MinecraftHandler(object):
    def __init__(self, video_shape=(512, 512), depth=3, ports=range(9000,9010)) -> None:
        self.host = MalmoPython.AgentHost()
        self.client_pool = MalmoPython.ClientPool()
        self.video_shape = video_shape
        self.depth = depth

        for port in ports:
            client_info = MalmoPython.ClientInfo('127.0.0.1', port)
            self.client_pool.add(client_info)

        self.state = None
        self.is_running = False
        self.num_obs = 0
        self.num_frames = 0

        self.can_breathe = True
        self.buried = False

        self.latest_obs = None
        self.latest_frame = None


    def get_state(self):
        self.state = self.host.getWorldState()
        self.is_running = self.state.is_mission_running
        self.num_obs = self.state.number_of_observations_since_last_state
        self.num_frames = self.state.number_of_video_frames_since_last_state

        if self.num_obs > 0:
            self.latest_obs = json.loads(self.state.observations[-1].text)
            if 'agentBlocks' not in self.latest_obs:
                self.buried = True
                self.can_breathe = False
            else:
                agent_block_occupants = self.latest_obs['agentBlocks']
                self.buried = agent_block_occupants[0] not in ['water', 'tallgrass', 'air']
                self.can_breathe = agent_block_occupants[1] in ['tallgrass', 'air']

        if self.num_frames > 0:
            image = np.frombuffer(self.state.video_frames[-1].pixels, dtype=np.uint8)
            image = image.reshape(self.video_shape+(self.depth,))
            self.latest_frame = tfunc.to_tensor(image).unsqueeze(0)

    def start_mission(self, mission_file, seed=b''):
        self.host.sendCommand("quit")
        self.get_state()
        while self.is_running:
            self.get_state()
        mission_xml = Path(mission_file).read_text()
        mission_xml = mission_xml.replace("SEED", str(seed).replace('"', "''"))
        mission_spec = MalmoPython.MissionSpec(mission_xml, True)
        mission_rec_spec = MalmoPython.MissionRecordSpec()
        self.host.startMission(mission_spec, 
                               self.client_pool,
                               mission_rec_spec,
                               0,
                               "STEVE")
        self.get_state()
        while not (self.is_running and self.num_obs > 0 and self.num_frames > 0):
            time.sleep(0.01)
            self.get_state()
        
        # if not self.can_breathe:
        #     raise Exception("Agent")
        while self.buried:
            self.host.sendCommand("jump 1")
            self.get_state()
        self.sendCommand("jump 0")
    
    def get_image(self):
        tries  = 0
        self.get_state()
        if not self.is_running:
            raise Exception("No mission running")
        while (self.num_obs < 1 or self.num_frames < 1) and tries < 3:
            time.sleep(0.1)
            self.get_state()
            tries += 1
        if (self.num_obs < 1 or self.num_frames < 1):
            raise Exception("No observation or image found")

        return self.latest_frame

    def send_command(self, command):
        self.host.sendCommand(command)
    
    def sendCommand(self, command):
        self.host.sendCommand(command)