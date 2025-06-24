import numpy as np
import torch
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from simple_dqn import QNetwork  

def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_env = gym.make("PongNoFrameskip-v4", render_mode = 'human')


    env = AtariPreprocessing(
        raw_env,
        frame_skip=4,        
        screen_size=84,     
        grayscale_obs=True, 
        scale_obs=False
    )
    env = FrameStack(env, num_stack=4)

    n_actions = env.action_space.n 
    in_channels = 4 
    q_net = QNetwork(in_channels, n_actions).to(device)
    q_net.load_state_dict(torch.load("dqn_pong_final.pth", map_location=device))
    q_net.eval()

    obs, _ = env.reset()
    state = np.array(obs)
    done = False

    while True:

        raw_env.render()

        state_t = (
            torch.from_numpy(state.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(device)
        )  


        with torch.no_grad():
            q_values = q_net(state_t)
            action = int(q_values.argmax(dim=1))  

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = np.array(next_obs)

        if done:
            obs, _ = env.reset()
            state = np.array(obs)


if __name__ == "__main__":
    play()
