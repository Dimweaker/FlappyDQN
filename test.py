import argparse
import torch
from src.game import FlappyBird
from tensorboardX import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--log_path", type=str, default="tensorboard")

    args = parser.parse_args()
    return args


def test(opt, model_name):
    writer = SummaryWriter(opt.log_path)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model = torch.load(opt.saved_path + model_name)
    game = FlappyBird()
    game.reset()
    game_times = 0
    max_score = 0
    average_score = 0
    image, reward, terminal = game.next_frame(0)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_image, reward, terminal = game.next_frame(action)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        if terminal:
            game_times += 1
            max_score = max(max_score, game.score)
            average_score = (average_score * (game_times - 1) + game.score) / game_times
            writer.add_scalar('score', game.score, game_times)
            writer.add_scalar('max_score', max_score, game_times)
            writer.add_scalar('average_score', average_score, game_times)
            game.reset()
            state, _, _ = game.next_frame(0)
            state = torch.from_numpy(state)
            if torch.cuda.is_available():
                state = state.cuda()
            state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
        else:
            state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt, opt.model_name)
