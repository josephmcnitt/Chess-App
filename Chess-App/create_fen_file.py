import my_chess
import sys
import os
import chess.pgn

pgn_folder = "Games"
output_file = "output.txt"

with open(output_file, "w") as f_out:
    for pgn_file in os.listdir(pgn_folder):
        with open(os.path.join(pgn_folder, pgn_file)) as f_in:
            while True:
                try:
                    game = chess.pgn.read_game(f_in)
                    if game is None:
                        break

                    board = game.board()
                    node = game
                    while not node.is_end():
                        node = node.variations[0]
                        try:
                            board.push(node.move)
                        except ValueError:
                            print(f"Illegal move in {pgn_file}, skipping...")
                            break
                        fen = board.fen()
                        comment = node.comment.strip().replace('\n', '') if node.comment else ""
                        if fen and not len(comment) == 0:
                            f_out.write(f"FEN: {fen}; Commentary: {comment}\n")
                except ValueError:
                    print(f"Error in {pgn_file}, skipping...")
                    break
