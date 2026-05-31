I borrowed the game from this guy: https://github.com/techwithtim/Python-Checkers/tree/master

Note for position arguements, I'll follow matrix notation such that

   White Pieces 
(0,0), (0,1), (0,2), (0,3)   # there are 4 possible squares per row
(1,0), (1,1), (1,2), (1,3)
(2,0), (2,1), (2,2), (2,3)   # down and right are positive 
(3,0), (3,1), (3,2), (3,3)
...
(7,0), (7,1), (7,2), (7,3)
   Black Pieces



the full board also goes

White
W, B, W, B, W, B, ...
B, W, B, W, B, W, ... where pieces are on the black tiles
...
Black (technically red)