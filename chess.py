import sys
print(sys.path)
sys.path.append('C:\\users\\jmmag\\appdata\\local\\programs\\python\\python310\\lib\\site-packages')

#We define a ChessBoard as the elements in an FEN:
# a position, the marker of whether it is white or black to play, whether there are castling rights, if there is an en-passantable
# square, halfmove clock, and the fullmove number.
class ChessBoard:
    def __init__(self, fen=None):
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
        self.player = 'w'
        self.castle = 'KQkq'
        self.enpassant = '-'
        self.halfmoves = '0'
        self.fullmoves = '1'

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
        fen += " " + self.player
        fen += " " + self.castle
        fen += " " + self.enpassant
        fen += " " + self.halfmoves
        fen += " " + self.fullmoves
        return fen

    def fen_to_position(self, fen: str) -> list[list[str]]:
        """
        Converts an FEN string to a board position represented as a 2D list.

        Args:
            fen (str): The FEN string representing the board position.

        Returns:
            List[List[str]]: A 2D list representing the board position.
        """
        rows = fen.split("/")

        # Initialize an empty board
        board = [["" for _ in range(8)] for _ in range(8)]

        # Loop through the FEN string to fill the board
        row_index = 0
        col_index = 0
        for char in "".join(rows):
            if char.isnumeric():
                col_index += int(char)
                if col_index > 7:
                    col_index = 0
                    row_index += 1
            else:
                if col_index > 7 or row_index > 7:
                    quit
                else:
                    print(row_index)
                    print(col_index)
                    board[row_index][col_index] = char
                    col_index += 1
                if col_index > 7:
                    col_index = 0
                    row_index += 1

        return board

    def print_board(self):
        for row in self.board:
            print(" ".join(row))
    
    letters_to_matrix = {
        '.':0
        ,'p':-1
        ,'P':1
        ,'n':-2
        ,'N':2
        ,'b':-3
        ,'B':3
        ,'r':-4
        ,'R':4
        ,'q':-5
        ,'Q':5
        ,'k':-6
        ,'K':6
    }

class ChessMoves:
    # position is a tuple
    def __init__(self, board, position):
        self.piece = board[position[0]][position[1]]
        self.moves = []
    def pawn_moves(self, board, position, moves):
        i = position(0)
        j = position(1)
        sgn = board.letters_to_matrix(self.piece) #This gives the sign to multiply for black vs white
        start = 3.5 - 2.5*sgn
        if i == start:  
            moves.append((i+1*sgn, j))
            moves.append((i+2*sgn, j))
        elif i > start:
            moves.append((i+1*sgn, j))
    def knight_moves(board, position, moves=[]):
        i = position(0)
        j = position(1)
        1
    def bishop_moves(board, position, moves=[]):
        i = position(0)
        j = position(1)
        1
    def rook_moves(self, position: tuple[int, int], board, moves):
        for i, j in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = position
            while True:
                x += i
                y += j
                if not self.is_valid_square((x, y)):
                    break
                piece = board[x][y]
                if piece == ".":
                    moves.append(f"{position[0]}{position[1]}{x}{y}")
                elif piece.isupper() != board[position[0]][position[1]].isupper():
                    moves.append(f"{position[0]}{position[1]}{x}{y}")
                    break
                else:
                    break
        return moves
    def queen_moves(board, position, moves=[]):
        i = position(0)
        j = position(1)
        1
    def king_moves(board, position, moves=[]):
        i = position(0)
        j = position(1)
        1
    
    def generate_moves(board, position):
        # Code to generate all possible moves for the current board state
        moves = []
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
                if piece == 1:
                    value += 1
                elif piece == -1:
                    value -= 1
                elif piece == 4:
                    value += 5
                elif piece == -4:
                    value -= 5
                elif piece == 2:
                    value += 3
                elif piece == -2:
                    value -= 3
                elif piece == 3:
                    value += 3
                elif piece == -3:
                    value -= 3
                elif piece == 5:
                    value += 9
                elif piece == -5:
                    value -= 9
                elif piece == 6:
                    value += 100
                elif piece == -6:
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
    
    def pgn_to_move(self, pgn: str, board) -> str:
        # Assuming player is white for now.
        # Get the piece letter from the PGN move
        if pgn[0].islower():
            piece = 1
        elif pgn[0] == '0':
            piece = 6
        else: piece = pgn[0]
        # Get the ending square of the move from the PGN move
        if pgn == '0-0':
            end_square = 'g1'
        elif pgn == '0-0-0':
            end_square = 'c1'
        else: end_square = pgn[-2:]
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
                    print(moves)
                    print(end_square)
                    if moves == None:
                        quit
                    elif end_square in moves:
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
class ChessGame:
    def start_game():
        print([
            [4, 2, 3, 5, 6, 3, 2, 4],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-4, -2, -3, -5, -6, -3, -2, -4],
            ])
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





