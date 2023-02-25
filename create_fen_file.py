import my_chess
import sys
sys.path.append('C:\\users\\jmmag\\appdata\\local\\programs\\python\\python310\\lib\\site-packages')
import chess.pgn

pgn_file = "StudyPlan.pgn"
output_file = "output.txt"

with open(pgn_file) as f_in, open(output_file, "w") as f_out:
    while True:
        game = chess.pgn.read_game(f_in)
        if game is None:
            break

        board = game.board()
        node = game
        while not node.is_end():
            node = node.variations[0]
            board.push(node.move)
            fen = board.fen()
            comment = node.comment.strip() if node.comment else ""
            if fen:
                f_out.write(f"FEN: {fen}; Commentary: {comment}\n")
