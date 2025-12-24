import argparse
import sys
from game_board_subclasses import TictactoeBoard, Connect4Board
from Q_Learning_AI_Tictactoe import TictactoeQLearning
from Q_Learning_AI_Connect4 import Connect4QLearning
from Minimax import MinimaxTictactoe

def run_training(game_type, episodes):
    print(f"Initialising training for {game_type}...")
    
    if game_type == "tictactoe":
        ai = TictactoeQLearning()
        board_cls = TictactoeBoard
    else:
        ai = Connect4QLearning()
        board_cls = Connect4Board

    win_count = 0
    draw_count = 0

    for i in range(1, episodes + 1):
        game = board_cls(player1=ai, player2=ai)
        result = game.start_game(mode="train", verbose=False, epsilon=0.2)
        
        if result["winner"] != 0: win_count += 1
        else: draw_count += 1

        if i % (episodes // 10) == 0:
            print(f"Episode {i}/{episodes} | Win Rate: {win_count/i:.2f} | Draws: {draw_count}")

    model_path = f"{game_type}_model.pkl"
    ai.save_model(model_path)
    print(f"Training complete. Model saved to {model_path}")

def run_benchmark(game_type, model_path, rounds=100):
    print(f"Benchmarking {game_type} against Optimal Baseline...")
    
    if game_type == "tictactoe":
        ai = TictactoeQLearning(pretrained_q_memory=model_path)
        opponent = MinimaxTictactoe()
        board_cls = TictactoeBoard
    else:
        print("Benchmarking Connect4 against Minimax not yet implemented.")
        return

    results = {"AI": 0, "Opponent": 0, "Draw": 0}
    for _ in range(rounds):
        game = board_cls(player1=ai, player2=opponent)
        res = game.start_game(mode="play", verbose=False)
        if res["winner"] == ai: results["AI"] += 1
        elif res["winner"] == 0: results["Draw"] += 1
        else: results["Opponent"] += 1

    print(f"Results after {rounds} rounds: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Game Agent Framework")
    parser.add_argument("--game", choices=["tictactoe", "connect4"], required=True)
    parser.add_argument("--mode", choices=["train", "benchmark", "play"], default="play")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--model", type=str, help="Path to saved model")

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args.game, args.episodes)
    elif args.mode == "benchmark":
        if not args.model:
            print("Error: --model path required for benchmarking.")
        else:
            run_benchmark(args.game, args.model)
