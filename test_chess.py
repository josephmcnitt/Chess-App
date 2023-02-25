import chess
board = chess.ChessBoard()
fen = board.position_to_fen()
print(board.board)
print(fen)
print(board.fen_to_position(fen))
moves = chess.ChessMoves(board,(1,1))
pawn_moves = moves.pawn_moves()