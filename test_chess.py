import chess
board = chess.ChessBoard()
fen = board.position_to_fen()
print(board.board)
print(fen)
print(board.fen_to_position(fen))