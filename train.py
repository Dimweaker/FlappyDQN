import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import random
from src.network import DeepQNetwork
from src.game import FlappyBird
import pandas as pd
import pickle
from tensorboardX import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")

    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=256, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.5)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_iters", type=int, default=2500000)
    parser.add_argument("--replay_memory_size", type=int, default=20000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="models")

    args = parser.parse_args()
    return args


def train(opt, iter=0):
    data = pd.read_csv('data/data.csv')
    writer = SummaryWriter(opt.log_path)

    max_score = data['max_score'].max()
    total_score = data['total_score'].iloc[-1]
    game_times = data['game_times'].iloc[-1]

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if iter:
        model = torch.load(opt.saved_path + '/model_{}.pth'.format(iter))
        replay_memory = pickle.load(open('data/replay_memory.pickle', 'rb'))
    else:
        model = DeepQNetwork()
        replay_memory = []
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game = FlappyBird()

    game.reset()
    image, reward, terminal = game.next_frame(0)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while iter < opt.num_iters:
        prediction = model(state)[0]
        if max_score < 50:
            epsilon = 0.5 * np.exp(-3 * np.log(10) / 2500000 * iter)
        elif max_score < 500:
            epsilon = 1e-3
        elif max_score < 1000:
            epsilon = 1e-4
        else:
            epsilon = 1e-6

        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(prediction).item()
        next_image, reward, terminal = game.next_frame(action)

        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        replay_memory.append((state, action, reward, next_state, terminal))
        if game.score:
            replay_memory.append((state, action, reward, next_state, terminal))
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[:len(replay_memory) - opt.replay_memory_size]

        batch = random.sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        state_batch = torch.cat(state_batch)
        next_state_batch = torch.cat(next_state_batch)
        action_batch = torch.from_numpy(np.array([[0, 1] if action else [1, 0] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])

        if torch.cuda.is_available():
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_batch = state_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        print("Iteration: {}, Loss: {}, epsilon:{}, reward:{}".format(iter, loss.item(), epsilon, reward))
        writer.add_scalar('loss', loss.item(), iter)
        writer.add_scalar('epsilon', epsilon, iter)
        writer.add_scalar('reward', reward, iter)
        writer.add_scalar('score', game.score, iter)
        writer.add_scalar('max_score', max_score, iter)
        writer.add_scalar('total', total_score, iter)
        writer.add_scalar('game_times', game_times, iter)

        if terminal:
            total_score += game.score
            max_score = max(max_score, game.score)
            game.reset()
            state, _, _ = game.next_frame(0)
            state = torch.from_numpy(state)
            if torch.cuda.is_available():
                state = state.cuda()
            state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
            game_times += 1
        else:
            state = next_state

        iter += 1

        if iter % 50000 == 0:
            torch.save(model, os.path.join(opt.saved_path, "model_{}.pth".format(iter)))
            print("Model saved")
            data = data.append({'iter': iter, 'loss': loss.item(), 'epsilon': epsilon, 'total_score': total_score,
                                'max_score': max_score, 'game_times': game_times}, ignore_index=True)
            data.to_csv('data/data.csv', index=False)
            with open('data/replay_memory.pickle', 'wb') as f:
                pickle.dump(replay_memory, f)

    torch.save(model, os.path.join(opt.saved_path, "model_final.pth"))
    print("Model saved")


if __name__ == "__main__":
    opt = get_args()
    train(opt, opt.iter)
