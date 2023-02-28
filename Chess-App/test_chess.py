import my_chess
import create_fen_file
board = my_chess.ChessBoard()
fen = board.position_to_fen()
print(board.board)
print(fen)
print(board.fen_to_position(fen))
moves = my_chess.ChessMoves(board,(1,1))
pawn_moves = moves.pawn_moves()

