import math

class ChessBoard:
    def __init__(self):
        starting_position = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        self.board = starting_position


    def print_board(self):
        for row in self.board:
            print(" ".join(row))
    
    def generate_moves(self):
        # Code to generate all possible moves for the current board state
        moves = []
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece == 'p':
                    # white pawn
                    if i == 1:
                        # pawn's first move
                        moves.append((i+1, j))
                        moves.append((i+2, j))
                    elif i > 1:
                        moves.append((i+1, j))
                        # pawn captures
                        if j > 0:
                            if self.board[i+1][j-1].islower():
                                moves.append((i+1, j-1))
                        if j < 7:
                            if self.board[i+1][j+1].islower():
                                moves.append((i+1, j+1))
                elif piece == 'P':
                    # black pawn
                    if i == 6:
                        # pawn's first move
                        moves.append((i-1, j))
                        moves.append((i-2, j))
                    elif i < 6:
                        moves.append((i-1, j))
                        # pawn captures
                        if j > 0:
                            if self.board[i-1][j-1].isupper():
                                moves.append((i-1, j-1))
                        if j < 7:
                            if self.board[i-1][j+1].isupper():
                                moves.append((i-1, j+1))
                elif piece == 'r':
                    # white rook
                    # horizontal moves to the left
                    for k in range(j-1, -1, -1):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                        elif self.board[i][k].islower():
                            moves.append((i, k))
                            break
                        else:
                            break
                    # horizontal moves to the right
                    for k in range(j+1, 8):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                            break
                        else:
                            break
                    # vertical moves up
                    for k in range(i-1, -1, -1):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].islower():
                            moves.append((k, j))
                            break
                        else:
                            break
                    # vertical moves down
                    for k in range(i+1, 8):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].islower():
                            moves.append((k, j))
                            break
                        else:
                            break
                elif piece == 'R':
                    # black rook
                    # horizontal moves to the left
                    for k in range(j-1, -1, -1):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                        elif self.board[i][k].isupper():
                            moves.append((i, k))
                            break
                        else:
                            break
                    # horizontal moves to the right
                    for k in range(j+1, 8):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                        elif self.board[i][k].isupper():
                            moves.append((i, k))
                            break
                        else:
                            break
                    # vertical moves up
                    for k in range(i-1, -1, -1):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].isupper():
                            moves.append((k, j))
                            break
                        else:
                            break
                    # vertical moves down
                    for k in range(i+1, 8):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].isupper():
                            moves.append((k, j))
                            break
                        else:
                            break
                elif piece == 'n':
                    # white knight
                    if i > 1:
                        if j > 0:
                            if self.board[i-2][j-1].islower():
                                moves.append((i-2, j-1))
                        if j < 7:
                            if self.board[i-2][j+1].islower():
                                moves.append((i-2, j+1))
                    if i > 0:
                        if j > 1:
                            if self.board[i-1][j-2].islower():
                                moves.append((i-1, j-2))
                        if j < 6:
                            if self.board[i-1][j+2].islower():
                                moves.append((i-1, j+2))
                    if i < 7:
                        if j > 0:
                            if self.board[i+2][j-1].islower():
                                moves.append((i+2, j-1))
                        if j < 7:
                            if self.board[i+2][j+1].islower():
                                moves.append((i+2, j+1))
                    if i < 6:
                        if j > 1:
                            if self.board[i+1][j-2].islower():
                                moves.append((i+1, j-2))
                        if j < 6:
                            if self.board[i+1][j+2].islower():
                                moves.append((i+1, j+2))
                elif piece == 'N':
                    # black knight
                    if i > 1:
                        if j > 0:
                            if self.board[i-2][j-1].isupper():
                                moves.append((i-2, j-1))
                        if j < 7:
                            if self.board[i-2][j+1].isupper():
                                moves.append((i-2, j+1))
                    if i > 0:
                        if j > 1:
                            if self.board[i-1][j-2].isupper():
                                moves.append((i-1, j-2))
                        if j < 6:
                            if self.board[i-1][j+2].isupper():
                                moves.append((i-1, j+2))
                    if i < 7:
                        if j > 0:
                            if self.board[i+2][j-1].isupper():
                                moves.append((i+2, j-1))
                    if j < 7:
                        if i < 7:
                            if self.board[i+2][j+1].isupper():
                                moves.append((i+2, j+1))
                if i < 6:
                    if j > 1:
                        if self.board[i+1][j-2].isupper():
                            moves.append((i+1, j-2))
                    if j < 6:
                        if self.board[i+1][j+2].isupper():
                            moves.append((i+1, j+2))
                elif piece == 'b':
                    # white bishop
                    # diagonal moves to the top-left
                    for k in range(1, min(8-i, j)+1):
                        if self.board[i+k][j-k] == '.':
                            moves.append((i+k, j-k))
                        elif self.board[i+k][j-k].islower():
                            moves.append((i+k, j-k))
                            break
                        else:
                            break
                    # diagonal moves to the top-right
                    for k in range(1, min(8-i, 8-j)+1):
                        if self.board[i+k][j+k] == '.':
                            moves.append((i+k, j+k))
                        elif self.board[i+k][j+k].islower():
                            moves.append((i+k, j+k))
                            break
                        else:
                            break
                elif piece == 'B':
                    # black bishop
                    # diagonal moves to the bottom-left
                    for k in range(1, min(i+1, j)+1):
                        if self.board[i-k][j-k] == '.':
                            moves.append((i-k, j-k))
                        elif self.board[i-k][j-k].isupper():
                            moves.append((i-k, j-k))
                            break
                        else:
                            break
                    # diagonal moves to the bottom-right
                    for k in range(1, min(i+1, 8-j)+1):
                        if self.board[i-k][j+k] == '.':
                            moves.append((i-k, j+k))
                        elif self.board[i-k][j+k].isupper():
                            moves.append((i-k, j+k))
                            break
                        else:
                            break
                elif piece == 'q':
                    # white queen moves
                    # diagonal moves to the top-left
                    for k in range(1, min(i, j) + 1):
                        if self.board[i-k][j-k] == '.':
                            moves.append((i-k, j-k))
                        elif self.board[i-k][j-k].islower():
                            moves.append((i-k, j-k))
                            break
                        else:
                            break
                    # diagonal moves to the top-right
                    for k in range(1, min(i, 8-j-1) + 1):
                        if self.board[i-k][j+k] == '.':
                            moves.append((i-k, j+k))
                        elif self.board[i-k][j+k].islower():
                            moves.append((i-k, j+k))
                            break
                        else:
                            break
                    # diagonal moves to the bottom-left
                    for k in range(1, min(8-i-1, j) + 1):
                        if self.board[i+k][j-k] == '.':
                            moves.append((i+k, j-k))
                        elif self.board[i+k][j-k].islower():
                            moves.append((i+k, j-k))
                            break
                        else:
                            break
                    # diagonal moves to the bottom-right
                    for k in range(1, min(8-i-1, 8-j-1) + 1):
                        if self.board[i+k][j+k] == '.':
                            moves.append((i+k, j+k))
                        elif self.board[i+k][j+k].islower():
                            moves.append((i+k, j+k))
                            break
                        else:
                            break
                    # vertical moves
                    for k in range(i-1, -1, -1):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].islower():
                            moves.append((k, j))
                            break
                        else:
                            break
                    for k in range(i+1, 8):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].islower():
                            moves.append((k, j))
                            break
                        else:
                            break
                    # horizontal moves
                    for k in range(j-1, -1, -1):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                        elif self.board[i][k].islower():
                            moves.append((i, k))
                            break
                        else:
                            break
                    for k in range(j+1, 8):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                        elif self.board[i][k].islower():
                            moves.append((i, k))
                            break
                        else:
                            break
                elif piece == 'Q':
                    # Black queen
                    # diagonal moves to the top left
                    i_, j_ = i, j
                    while i_ > 0 and j_ > 0:
                        i_ -= 1
                        j_ -= 1
                        if self.board[i_][j_] == '.':
                            moves.append((i_, j_))
                        elif self.board[i_][j_].isupper():
                            moves.append((i_, j_))
                            break
                        else:
                            break
                    # diagonal moves to the top right
                    i_, j_ = i, j
                    while i_ > 0 and j_ < 7:
                        i_ -= 1
                        j_ += 1
                        if self.board[i_][j_] == '.':
                            moves.append((i_, j_))
                        elif self.board[i_][j_].isupper():
                            moves.append((i_, j_))
                            break
                        else:
                            break
                    # diagonal moves to the bottom left
                    i_, j_ = i, j
                    while i_ < 7 and j_ > 0:
                        i_ += 1
                        j_ -= 1
                        if self.board[i_][j_] == '.':
                            moves.append((i_, j_))
                        elif self.board[i_][j_].isupper():
                            moves.append((i_, j_))
                            break
                        else:
                            break
                    # diagonal moves to the bottom right
                    i_, j_ = i, j
                    while i_ < 7 and j_ < 7:
                        i_ += 1
                        j_ += 1
                        if self.board[i_][j_] == '.':
                            moves.append((i_, j_))
                        elif self.board[i_][j_].isupper():
                            moves.append((i_, j_))
                            break
                        else:
                            break
                    # horizontal moves to the left
                    for k in range(j-1, -1, -1):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                        elif self.board[i][k].isupper():
                            moves.append((i, k))
                            break
                        else:
                            break
                    # horizontal moves to the right
                    for k in range(j+1, 8):
                        if self.board[i][k] == '.':
                            moves.append((i, k))
                        elif self.board[i][k].isupper():
                            moves.append((i, k))
                            break
                        else:
                            break
                    # vertical moves up
                    for k in range(i-1, -1, -1):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].isupper():
                            moves.append((k, j))
                            break
                        else:
                            break
                    # vertical moves down
                    for k in range(i+1, 8):
                        if self.board[k][j] == '.':
                            moves.append((k, j))
                        elif self.board[k][j].isupper():
                            moves.append((k, j))
                            break
                        else:
                            break
                elif piece == 'k':
                    for x_offset in range(-1, 2):
                        for y_offset in range(-1, 2):
                            if x_offset == 0 and y_offset == 0:
                                continue
                            x, y = i + x_offset, j + y_offset
                            if 0 <= x < 8 and 0 <= y < 8:
                                if self.board[x][y].islower():
                                    moves.append((x, y))
                elif piece == 'K':
                    for x_offset in range(-1, 2):
                        for y_offset in range(-1, 2):
                            if x_offset == 0 and y_offset == 0:
                                continue
                            x, y = i + x_offset, j + y_offset
                            if 0 <= x < 8 and 0 <= y < 8:
                                if self.board[x][y].isupper():
                                    moves.append((x, y))
    def minimax(self, depth, is_maximizing):
        MAX_DEPTH = 5
        if depth == 0:
            return self.evaluate_position()
        
        best_score = None
        for move in self.generate_moves():
            self.make_move(move)
            score = self.minimax(depth - 1, not is_maximizing)
            self.undo_move(move)
            if is_maximizing:
                if best_score is None or score > best_score:
                    best_score = score
                    if depth == MAX_DEPTH:
                        self.best_move = move
                        self.best_score = score
            else:
                if best_score is None or score < best_score:
                    best_score = score
        return best_score
    def evaluate_position(self):
        value = 0
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece == 'p':
                    value += 1
                elif piece == 'P':
                    value -= 1
                elif piece == 'r':
                    value += 5
                elif piece == 'R':
                    value -= 5
                elif piece == 'n':
                    value += 3
                elif piece == 'N':
                    value -= 3
                elif piece == 'b':
                    value += 3
                elif piece == 'B':
                    value -= 3
                elif piece == 'q':
                    value += 9
                elif piece == 'Q':
                    value -= 9
                elif piece == 'k':
                    value += 100
                elif piece == 'K':
                    value -= 100
        return value

    def make_move(self, move):
        start, end = move
        piece = self.board[start[0]][start[1]]
        self.board[end[0]][end[1]] = piece
        self.board[start[0]][start[1]] = '.'
    def undo_move(self, move):
        # Reverse the actions taken in make_move
        i, j = move[0]
        piece = self.board[i][j]
        self.board[i][j] = self.board[move[1][0]][move[1][1]]
        self.board[move[1][0]][move[1][1]] = piece


    # def undo_move(self, move):
    #     # Code to undo the given move and return the board state to its previous state
    def alphabeta(self, depth, alpha, beta, is_maximizing):
        if depth == 0:
            return self.evaluate_position()
        if is_maximizing:
            best_score = -float('inf')
            for move in self.generate_moves():
                self.make_move(move)
                score = self.alphabeta(depth - 1, alpha, beta, False)
                self.undo_move(move)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        else:
            best_score = float('inf')
            for move in self.generate_moves():
                self.make_move(move)
                score = self.alphabeta(depth - 1, alpha, beta, True)
    def find_best_move(self):
        MAX_DEPTH = 4
        self.minimax(MAX_DEPTH, True)
        return self.best_move
    def move_to_pgn(self, move, board):
        # Get the starting and ending coordinates of the move
        start = move[:2]
        end = move[2:]
        # Get the piece that is being moved
        piece = board[start[0]][start[1]]
        # Initialize the PGN move string
        pgn_move = ""
        # Add the piece letter to the PGN move
        pgn_move += piece.upper()
        # Get all the pieces that can move to the end position
        attacking_pieces = []
        for i, row in enumerate(board):
            for j, square in enumerate(row):
                if square != "." and square.upper() == piece.upper():
                    moves = self.generate_moves((i, j), board)
                    if end in moves:
                        attacking_pieces.append((i, j))
        # If there is more than one attacking piece, add the file/rank of the starting square to disambiguate
        if len(attacking_pieces) > 1:
            if piece.isupper():
                file_or_rank = chr(ord("a") + start[1])
            else:
                file_or_rank = str(8 - start[0])
            pgn_move += file_or_rank
        # Add the "x" if the move captures a piece
        if board[end[0]][end[1]] != ".":
            pgn_move += "x"
        # Add the ending square of the move
        if piece.isupper():
            pgn_move += chr(ord("a") + end[1]) + str(8 - end[0])
        else:
            pgn_move += chr(ord("a") + end[1]) + str(end[0] + 1)
        return pgn_move
    def position_to_fen(self):
        fen = ""
        empty_count = 0
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece == ".":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen += piece
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            if i < 7:
                fen += "/"
        fen += " "
        fen += "w" if self.white_to_move else "b"
        fen += " "
        fen += "- "
        fen += "0 1"
        return fen
    def pgn_to_move(self, pgn: str, board) -> str:
        # Get the piece letter from the PGN move
        piece = pgn[0]
        # Get the ending square of the move from the PGN move
        end_square = pgn[-2:]
        # Get the file/rank of the starting square if it is included in the PGN move
        file_or_rank = None
        if len(pgn) > 3:
            file_or_rank = pgn[1]
        # Initialize a list to store the possible starting squares for the move
        possible_starts = []
        for i, row in enumerate(board):
            for j, square in enumerate(row):
                if square.upper() == piece.upper():
                    moves = self.generate_moves((i, j), board)
                    if end_square in moves:
                        possible_starts.append((i, j))
        # If there is only one possible starting square, return the move
        if len(possible_starts) == 1:
            start_square = possible_starts[0]
            return f"{start_square[0]}{start_square[1]}{end_square}"
        else:
            # If there is more than one possible starting square, use the file/rank from the PGN move to disambiguate
            for start_square in possible_starts:
                if file_or_rank == chr(ord("a") + start_square[1]) or file_or_rank == str(8 - start_square[0]):
                    return f"{start_square[0]}{start_square[1]}{end_square}"
        # If the PGN move is invalid, return None
        return None
def start_game():
    chess_board = ChessBoard()
    while True:
        # Print the current board
        print(chess_board)
        # Get input from the user in PGN format
        user_move = input("Enter your move: ")
        # Convert the user's input to the code's "move" notation
        move = chess_board.pgn_to_move(user_move, chess_board.board)
        # Check if the move is valid
        if move == None or move not in chess_board.generate_moves():
            print("Invalid move, please try again.")
            continue
        # Make the move on the board
        chess_board.make_move(move)
        # Print the move in PGN format
        print("Move: " + chess_board.move_to_pgn(move))
        # Check if the game is over
        if chess_board.is_game_over():
            print("Game over!")
            break

start_game()




